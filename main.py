import os
import math
import random
import numpy as np
import pandas as pd
from typing import List
from fastapi import Body
from random import sample
from fastapi import FastAPI
from datetime import datetime
from dotenv import load_dotenv
from typing import Union, Optional
from collections import OrderedDict
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Supabase Initialization
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Constants for table names
TWO_TOWER_TABLE = "temp_ik_resultsnudgetwscore"
INTERACTION_LOG_TABLE = "interaction_log"
REWARD_POLICY_LOG_TABLE = "reward_policy_log"
PERFORMANCE_LOG_TABLE = "performance_log"
AUTOMATION_USERS_TABLE = "automation_userswithriskmab"

# Utility function
def GaussianDistribution(loc, scale, size):
    return np.random.normal(loc, scale, size)

def interaction_log_data():
    response = supabase.table(INTERACTION_LOG_TABLE).select("*", count="exact").execute()
    return response

def interaction_log_data_bystate_id(state_id: str):
    response = supabase.table(INTERACTION_LOG_TABLE).select("*").eq("state_id", state_id).execute()
    return response.data

def cold_start_data():
    query = supabase.table(AUTOMATION_USERS_TABLE).select("*", count="exact")
    response = query.execute()
    return response

def two_tower_data():
    query = supabase.table(TWO_TOWER_TABLE).select("*", count="exact")
    response = query.execute()
    return response

@app.get("/product-recommendation/{state_id}")
def product_recommendation_by_state_id(state_id:str):
    # Data awal
    raw_data = cold_start_data().data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = interaction_log_data().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')
    
    # Inisialisasi dictionary untuk menyimpan statistik
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}
    cumulative_reward_log = []
    cumulative_regret_log = []
    optimal_arm_counts = []

    # Hitung rata-rata reward per produk
    avg_rewards_per_product = rewardFull.groupby('product_title')['OverallScore'].mean().to_dict()
    top5_avg_rewards = sorted(avg_rewards_per_product.values(), reverse=True)[:5]
    best_expected_reward = sum(top5_avg_rewards) / len(top5_avg_rewards)
    best_product = max(avg_rewards_per_product, key=avg_rewards_per_product.get)

    # Inisialisasi reward awal berdasarkan data historis
    users = list(rewardFull.user_id.unique())
    for id in users:
        subset = rewardFull[rewardFull['user_id'] == id]
        countDic[id] = {row['product_title']: int(row['OverallScore']) for _, row in subset.iterrows()}

    # Fungsi sampling produk menggunakan epsilon-greedy
    def sampProduct(nproducts, state_id, epsilon):
        sorted_policies = sorted(polDic[state_id].items(), key=lambda kv: kv[1], reverse=True)
        topProducts = [prod[0] for prod in sorted_policies[:nproducts]]
        seg_products = []

        if best_product in polDic[state_id] and best_product not in seg_products:
            seg_products.append(best_product)

        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                next_prod = topProducts.pop(0)
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    next_prod = sample(available_products, 1)[0]
                else:
                    break
            if next_prod not in seg_products:
                seg_products.append(next_prod)

        return list(OrderedDict.fromkeys(seg_products))

    # Mulai sesi interaksi
    if state_id not in countDic:
        countDic[state_id] = {}
    if state_id not in polDic:
        polDic[state_id] = {}
    if state_id not in rewDic:
        rewDic[state_id] = {}
    if state_id not in recoCount:
        recoCount[state_id] = {}

    previous_entries = [entry for entry in interaction_log if entry['state_id'] == state_id]
    is_new_user = len(previous_entries) == 0

    if not countDic[state_id]:
        for product in avg_rewards_per_product:
            countDic[state_id][product] = 0
            noise = np.random.normal(loc=0.0, scale=1.0)
            polDic[state_id][product] = avg_rewards_per_product[product] + noise
            rewDic[state_id][product] = 0
            recoCount[state_id][product] = 1

    for pkey in countDic[state_id].keys():
        if pkey not in polDic[state_id]:
            polDic[state_id][pkey] = GaussianDistribution(loc=countDic[state_id][pkey], scale=1, size=1)[0].round(2)
        if pkey not in rewDic[state_id]:
            rewDic[state_id][pkey] = GaussianDistribution(loc=countDic[state_id][pkey], scale=1, size=1)[0].round(2)

    nProducts = 5
    epsilon = 0.3 if is_new_user else 0.01

    if previous_entries:
        last_entry = previous_entries[-1]
        next_data = last_entry['next_recommended']
        try:
            seg_products = eval(next_data) if isinstance(next_data, str) else next_data
        except Exception as e:
            print(f"Gagal memuat rekomendasi sebelumnya: {e}")
            seg_products = []
        print(f"(Menggunakan rekomendasi lanjutan dari interaksi sebelumnya untuk state {state_id})")
    else:
        seg_products = sampProduct(nProducts, state_id, epsilon)

    products = supabase.table("nudge_product").select("*").in_("product_title", seg_products).execute()
    ordered_products = sorted(products.data, key=lambda x: seg_products.index(x["product_title"]))
    return ordered_products

@app.post("/product-recommendation/{state_id}")
def buy_product_by_state_id(state_id:str, buy_list: List[str] = Body(default=["Tabungan Haji"])):
    # Data awal
    raw_data = cold_start_data().data
    data = pd.DataFrame(raw_data)
    data['OverallScore'] = data['score']
    rewardFull = data.groupby(['user_id', 'product_title', 'content'])['OverallScore'].sum().reset_index()

    # Data interaction
    raw_interaction_log = interaction_log_data().data
    interaction_log = pd.DataFrame(raw_interaction_log).to_dict('records')
    
    # Inisialisasi dictionary untuk menyimpan statistik
    countDic, polDic, rewDic, recoCount = {}, {}, {}, {}
    cumulative_reward_log = []
    cumulative_regret_log = []
    optimal_arm_counts = []

    # Hitung rata-rata reward per produk
    avg_rewards_per_product = rewardFull.groupby('product_title')['OverallScore'].mean().to_dict()
    top5_avg_rewards = sorted(avg_rewards_per_product.values(), reverse=True)[:5]
    best_expected_reward = sum(top5_avg_rewards) / len(top5_avg_rewards)
    best_product = max(avg_rewards_per_product, key=avg_rewards_per_product.get)

    # Inisialisasi reward awal berdasarkan data historis
    users = list(rewardFull.user_id.unique())
    for id in users:
        subset = rewardFull[rewardFull['user_id'] == id]
        countDic[id] = {row['product_title']: int(row['OverallScore']) for _, row in subset.iterrows()}

    # Fungsi sampling produk menggunakan epsilon-greedy
    def sampProduct(nproducts, state_id, epsilon):
        sorted_policies = sorted(polDic[state_id].items(), key=lambda kv: kv[1], reverse=True)
        topProducts = [prod[0] for prod in sorted_policies[:nproducts]]
        seg_products = []

        if best_product in polDic[state_id] and best_product not in seg_products:
            seg_products.append(best_product)

        while len(seg_products) < nproducts:
            probability = np.random.rand()
            if probability >= epsilon and topProducts:
                next_prod = topProducts.pop(0)
            else:
                available_products = list(rewardFull['product_title'].unique())
                available_products = [p for p in available_products if p not in seg_products]
                if available_products:
                    next_prod = sample(available_products, 1)[0]
                else:
                    break
            if next_prod not in seg_products:
                seg_products.append(next_prod)

        return list(OrderedDict.fromkeys(seg_products))

    # Mulai sesi interaksi
    if state_id not in countDic:
        countDic[state_id] = {}
    if state_id not in polDic:
        polDic[state_id] = {}
    if state_id not in rewDic:
        rewDic[state_id] = {}
    if state_id not in recoCount:
        recoCount[state_id] = {}

    previous_entries = [entry for entry in interaction_log if entry['state_id'] == state_id]
    is_new_user = len(previous_entries) == 0

    if not countDic[state_id]:
        for product in avg_rewards_per_product:
            countDic[state_id][product] = 0
            noise = np.random.normal(loc=0.0, scale=1.0)
            polDic[state_id][product] = avg_rewards_per_product[product] + noise
            rewDic[state_id][product] = 0
            recoCount[state_id][product] = 1

    for pkey in countDic[state_id].keys():
        if pkey not in polDic[state_id]:
            polDic[state_id][pkey] = GaussianDistribution(loc=countDic[state_id][pkey], scale=1, size=1)[0].round(2)
        if pkey not in rewDic[state_id]:
            rewDic[state_id][pkey] = GaussianDistribution(loc=countDic[state_id][pkey], scale=1, size=1)[0].round(2)

    nProducts = 5
    epsilon = 0.3 if is_new_user else 0.01

    if previous_entries:
        last_entry = previous_entries[-1]
        next_data = last_entry['next_recommended']
        try:
            seg_products = eval(next_data) if isinstance(next_data, str) else next_data
        except Exception as e:
            print(f"Gagal memuat rekomendasi sebelumnya: {e}")
            seg_products = []
        print(f"(Menggunakan rekomendasi lanjutan dari interaksi sebelumnya untuk state {state_id})")
    else:
        seg_products = sampProduct(nProducts, state_id, epsilon)

    # Fungsi update policy & reward setelah interaksi
    def valueUpdater(seg_products, loc, custList, epsilon):
        reward_before = [rewDic[state_id].get(p, 0) for p in custList]
        policy_before = [polDic[state_id].get(p, 0) for p in custList]

        total_reward_this_round = 0.0
        regret_this_round = 0.0
        picked_best = False

        for prod in custList:
            if prod not in rewDic[state_id]:
                rewDic[state_id][prod] = 0.0
            if prod not in polDic[state_id]:
                polDic[state_id][prod] = 0.0
            if prod not in recoCount[state_id]:
                recoCount[state_id][prod] = 1

            rew = GaussianDistribution(loc=loc, scale=0.5, size=1)[0].round(2)
            rewDic[state_id][prod] += rew
            polDic[state_id][prod] += (1 / recoCount[state_id][prod]) * (rew - polDic[state_id][prod])
            recoCount[state_id][prod] += 1

            total_reward_this_round += rew
            expected_reward = avg_rewards_per_product.get(prod, 0)
            regret_this_round += max(0.0, best_expected_reward - expected_reward)

            if prod == best_product:
                picked_best = True

            epsilon = max(0.01, epsilon * 0.95)

        reward_after = [rewDic[state_id][p] for p in custList]
        policy_after = [polDic[state_id][p] for p in custList]
        next_recommended = sampProduct(nProducts, state_id, epsilon)

        interaction_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state_id": state_id,
            "product_offered": seg_products,
            "product_bought": custList,
            "reward_before": reward_before,
            "policy_before": policy_before,
            "reward_after": reward_after,
            "policy_after": policy_after,
            "next_recommended": next_recommended
        }

        interaction_log.append(interaction_entry)
        cumulative_reward_log.append(total_reward_this_round)
        cumulative_regret_log.append(
            cumulative_regret_log[-1] + regret_this_round if cumulative_regret_log else regret_this_round
        )
        optimal_arm_counts.append(int(picked_best))

        supabase.table("interaction_log").insert(interaction_entry).execute()
        return next_recommended, epsilon
    
    seg_products, epsilon = valueUpdater(seg_products, 5, buy_list, epsilon)
    n_new = len(cumulative_reward_log)
    new_interactions = interaction_log[-n_new:] 

    reward_policy_df = pd.DataFrame({
        "product_title": list(rewDic[state_id].keys()),
        "updated_reward": list(rewDic[state_id].values()),
        "updated_policy": [polDic[state_id].get(p, 0) for p in rewDic[state_id].keys()],
        "state_id": state_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })    

    performance_df = pd.DataFrame({
        "timestamp": [entry["timestamp"] for entry in new_interactions],
        "state_id": [entry["state_id"] for entry in new_interactions],
        "total_reward": cumulative_reward_log,
        "cumulative_regret": cumulative_regret_log,
        "picked_best_arm": optimal_arm_counts,
        "picked_from_recommendation": [
            any(prod in entry["product_offered"] for prod in entry["product_bought"])
            for entry in new_interactions
        ]
    })

    reward_policy_data = reward_policy_df.to_dict(orient='records')
    supabase.table(REWARD_POLICY_LOG_TABLE).insert(reward_policy_data).execute()

    performance_data = performance_df.to_dict(orient="records")
    supabase.table(PERFORMANCE_LOG_TABLE).insert(performance_data).execute()

    return {
        "update_rewards":dict(sorted(rewDic[state_id].items(), key=lambda x: x[1], reverse=True)),
        "update_policies": dict(sorted(polDic[state_id].items(), key=lambda x: x[1], reverse=True))
    }