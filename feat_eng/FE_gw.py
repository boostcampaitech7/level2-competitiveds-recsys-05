#!/usr/bin/env python
# coding: utf-8

# # 0. Load Libraries

# In[1]:


import numpy as np
import pandas as pd
import os
import gc
import utils
import feat_eng
# import dr_clust

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[3]:


file_path = '../notebooks/data'
train = pd.read_csv(os.path.join(file_path, 'train.csv')).assign(_type = 'train')
test = pd.read_csv(os.path.join(file_path, 'test.csv')).assign(_type = 'test')


# #### train data 중복 데이터 제거

# In[4]:


print(f'중복 제거 전 train.shape = {train.shape}')
dup_train = train.drop('index', axis=1).duplicated()
train = train[~dup_train]
print(f'중복 제거 후 train.shape = {train.shape}')


# #### train, test data 병합

# In[5]:


df = pd.concat([train, test])
del train, test, dup_train
print(f'df.shape = {df.shape}')


# #### 계약 연도 변수 만들기

# In[6]:


df['contract_year'] = df['contract_year_month'].astype(str).apply(lambda x: x[:4]).astype(int).values


# #### 아파트 단지 ID 만들기

# In[7]:


df = feat_eng.create_complex_id(df=df, lat_col='latitude', lon_col='longitude')


# In[8]:


subwayInfo = pd.read_csv(os.path.join(file_path, 'subwayInfo.csv'))
schoolInfo = pd.read_csv(os.path.join(file_path, 'schoolinfo.csv'))
parkInfo = pd.read_csv(os.path.join(file_path, 'parkInfo.csv'))


# # 1. POI 데이터
# 
# ## 1.1 Distance to the Nearest POI (km)

# In[9]:


apt = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

apt = apt.assign(
    # 1. Distance to the nearest subway
    nearest_subway_distance_km = feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=subwayInfo)[0],
    
    # 2. Distance to the nearest (elementary, middel, high) school
    nearest_elementary_distance_km = feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=schoolInfo[schoolInfo['schoolLevel']=='elementary'][['latitude', 'longitude']])[0],
    nearest_middle_distance_km = feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=schoolInfo[schoolInfo['schoolLevel']=='middle'][['latitude', 'longitude']])[0],
    nearest_high_distance_km = feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=schoolInfo[schoolInfo['schoolLevel']=='high'][['latitude', 'longitude']])[0],
    
    # 3. Distance to the nearest park & Area of the nearest park
    nearest_park_distance_km = feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=parkInfo[['latitude', 'longitude']])[0],
    nearest_park_area = parkInfo['area'][feat_eng.nearest_POI(apt=apt[['latitude', 'longitude']], POI=parkInfo[['latitude', 'longitude']])[1]].values
)


# # 2. 특정 반경 안 POI 개수

# In[10]:


apt = apt.assign(
    # 1. 지하철역
    num_subway_within_0_5km = feat_eng.count_within_radius(apt, subwayInfo, radius_km=0.5),
    num_subway_within_1_5km = feat_eng.count_within_radius(apt, subwayInfo, radius_km=1.5),
    num_subway_within_3km = feat_eng.count_within_radius(apt, subwayInfo, radius_km=3),
    
    # 2. 학교
    num_elementary_within_1km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='elementary'], radius_km=1),
    num_elementary_within_3km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='elementary'], radius_km=3),
    num_middle_within_1km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='middle'], radius_km=1),
    num_middle_within_3km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='middle'], radius_km=3),
    num_high_within_1km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='high'], radius_km=1),
    num_high_within_3km = feat_eng.count_within_radius(apt, schoolInfo[schoolInfo['schoolLevel']=='high'], radius_km=3),
    
    # 3. 공원
    num_park_within_1km = feat_eng.count_within_radius(apt, parkInfo, radius_km=1),
    num_park_within_5km = feat_eng.count_within_radius(apt, parkInfo, radius_km=5)
)


# In[11]:


df = df.merge(apt, on=['latitude', 'longitude'], how='left')
del subwayInfo, schoolInfo, parkInfo, apt
gc.collect()


# # 3. 금리 데이터

# In[12]:


interestRate = pd.read_csv(os.path.join(file_path, 'interestRate.csv')).rename({'year_month': 'contract_year_month'}, axis=1)


# In[13]:


df = df.merge(feat_eng.add_interest_rate(interestRate), on='contract_year_month', how='left')
del interestRate

gc.collect()


# # 4. 신축 공급량과 비슷한 효과를 내도록 하는 age 관련 변수

# In[14]:


monthly_new_supply = feat_eng.add_new_supply_column(df, age_col='age', group_col='contract_year_month', supply_col_name='monthly_new_supply')
df = df.merge(monthly_new_supply, on='contract_year_month', how='left')

del monthly_new_supply


# # 5. 클러스터 변수 붙여주기
# 
# - `Standard_None_KMenas100_results.csv`:
#     - `['latittude', 'longitude']`
# - `Minmax_UMAP02_KMeans500_results.csv`:
#     - `['latitude', 'longitude', 'nearest_subway_distance_km', '2023_age', 'deposit_mean', 'deposit_std']`

# In[15]:


std_none_km100 = pd.read_csv("./cluster_data/Standard_None_KMeans100_results.csv")[['complex_id', 'Cluster']]
minmax_umap02_km100 = pd.read_csv("./cluster_data/Minmax_UMAP02_KMeans500_results.csv")[['complex_id', 'Cluster']]
selected_cols = ['latitude', 'longitude']


# In[16]:


# df = df.merge(std_none_km100, on='complex_id', how='left')
df = df.merge(minmax_umap02_km100, on='complex_id', how='left')


# # 6. 계약 건수

# In[17]:


df['contract_count_last_year'] = feat_eng.add_previous_contract_count(df, group_col = 'Cluster', grouping_type='year')
df['contract_count_last_month'] = feat_eng.add_previous_contract_count(df, group_col = 'Cluster', grouping_type='month')
df['contract_count_last_quarter'] = feat_eng.add_previous_contract_count(df, group_col = 'Cluster', grouping_type='quarter')
df['contract_count_last_half'] = feat_eng.add_previous_contract_count(df, group_col = 'Cluster', grouping_type='half')


# # 7. 전세가 FE

# 1. Train 데이터에 대해: 
#     - 먼저 train 데이터에서 `(Cluster, contract_year)`별 `deposit`의 평균과 표준편차를 계산
#     - 그 후, 각 행의 `contract_year`보다 이전에 존재하는 가장 최신의 값을 병합
#     - 이를 위해 먼저 `contract_year`별 누적된 통계를 만들어야 함
# 
# 2. Test 데이터에 대해:
#     - test 데이터의 `Cluster`가 train 데이터에 존재하는 경우 해당 `Cluster`의 값을 그대로 사용하고
#     - 그렇지 않으면 나머지 feature들과의 유클리드 거리를 계산하여 가장 가까운 `Cluster`를 찾아서 그 값을 사용

# In[18]:


# 이전 년도 클러스터별 전세가 평균, 표준편차 추가
df = feat_eng.calculate_deposit_stats(df, group_col='Cluster', time_unit='year')
# 이전 달 클러스터별 전세가 평균, 표준편차 추가
df = feat_eng.calculate_deposit_stats(df, group_col='Cluster', time_unit='month')
# 이전 분기 클러스터별 전세가 평균, 표준편차 추가
df = feat_eng.calculate_deposit_stats(df, group_col='Cluster', time_unit='quarter')
# 이전 (상/하)반기 클러스터별 전세가 평균, 표준편차 추가
df = feat_eng.calculate_deposit_stats(df, group_col='Cluster', time_unit='half')


# # 8. 층수, 면적 범주화

# In[20]:


df = feat_eng.categorize_floor(df)
df = feat_eng.categorize_area(df)


# In[24]:


df = utils.reduce_mem_usage(df, exclude_columns=['latitude', 'longitude'])


# In[22]:


df.to_csv(os.path.join(file_path, "fe_final_2.csv"), index=False)


# In[ ]:




