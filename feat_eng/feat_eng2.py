import pandas as pd
import numpy as np

# 파일 경로 및 파일 불러오기
data = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/JWORGINIZETEST/train_label.csv')
data = data[data['_type'] == 'train']
# data = data[data['contract_year_month'] == 202303]

data = data.groupby(['complex_id', 'latitude', 'longitude'], as_index=False).agg({
    'area_m2' : 'mean',
    'deposit' : 'mean',
    'built_year' : 'min'
})


# complex_id 기준으로 위도(latitude), 경도(longitude) 값만 추출
data = data.drop_duplicates(subset=['complex_id', 'area_m2'])
data = data.sort_values(by = ['latitude', 'longitude'])

# 데이터 형태 확인
print("데이터 형태:", data.shape)
print(data.head())  # 첫 5개 행 출력

# 사용할 변수 선택 
selected_columns = ['latitude', 'area_m2', 'built_year', 'longitude', 'deposit']

# 데이터 전처리 함수 호출
processed_data = select_and_preprocess_data(
    data=data,
    selected_columns=selected_columns,
    scaling_method='standard',  # 'standard' 또는 'minmax', 'robust'
    sample_size=None,           # 샘플링하지 않음
    random_state=42
)

print("전처리된 데이터의 형태:", processed_data.shape)

# 차원 축소 알고리즘 선택 및 파라미터 설정
dr_method = 'UMAP'
dr_params = {'n_components': 2}

# 차원 축소 적용
reduced_data, dr_used_params = apply_dimensionality_reduction(
    method=dr_method,
    data=processed_data,
    tune_hyperparameters=False,
    custom_params=dr_params,
    random_state=42
)

print(f"차원 축소된 데이터의 형태: {reduced_data.shape}")
print(f"사용된 차원 축소 파라미터: {dr_used_params}")

# 클러스터링 알고리즘 선택 및 파라미터 설정
cl_method = 'KMeans'
cl_params = {'n_clusters': 1000}

# 클러스터링 적용
model, labels, cl_used_params = apply_clustering(
    method=cl_method,
    data=reduced_data,
    tune_hyperparameters=False,
    custom_params=cl_params,
    random_state=42
)

print(f"클러스터 레이블의 형태: {labels.shape}")

print(f"사용된 클러스터링 파라미터: {cl_used_params}")

data2 = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/EDA&FE/final.csv')
data['cluster_labels2'] = labels
cluster_data = data[['cluster_labels2', 'complex_id']]

data3 = data2.merge(cluster_data, on=['complex_id'], how='left')
data3.to_csv('final2.csv', index = False)

import pandas as pd
from prophet import Prophet

data = pd.read_csv('./final2.csv')

# 최종 결과를 저장할 빈 데이터프레임 생성
final_results = pd.DataFrame()

# 1. 데이터를 cluster_labels2로 그룹화
grouped = data.groupby('cluster_labels2')

# 2. 각 클러스터별로 데이터 처리 및 예측
for cluster, group_data in grouped:
    # contract_year_month에서 연도와 월을 추출하여 반기 구분
    group_data['contract_year_month'] = group_data['contract_year_month'].astype(str)
    group_data['year'] = group_data['contract_year_month'].str[:4].astype(int)
    group_data['month'] = group_data['contract_year_month'].str[4:6].astype(int)
    
    # 반기 구분 (1~6월 상반기, 7~12월 하반기)
    group_data['half_year'] = group_data['month'].apply(lambda x: 'H1' if x <= 6 else 'H2')
    
    # 반기별 deposit 값 평균 계산
    semi_annual_data = group_data.groupby(['year', 'half_year']).agg({'deposit': 'mean'}).reset_index()
    
    # 반기를 실제 날짜로 변환하는 함수
    def half_year_to_date(row):
        if row['half_year'] == 'H1':  # 상반기 -> 해당 연도의 6월 30일로 설정
            return pd.Timestamp(year=row['year'], month=6, day=30)
        else:  # 하반기 -> 해당 연도의 12월 31일로 설정
            return pd.Timestamp(year=row['year'], month=12, day=31)

    # 'ds' 컬럼에 날짜 생성
    semi_annual_data['ds'] = semi_annual_data.apply(half_year_to_date, axis=1)
    semi_annual_data = semi_annual_data[['ds', 'deposit']].rename(columns={'deposit': 'y'})
    
    # Prophet 모델로 예측
    model = Prophet()
    model.fit(semi_annual_data)

    # 2024년 상반기 예측 (한 시점 예측)
    future = model.make_future_dataframe(periods=1, freq='6M')
    forecast = model.predict(future)
    predicted_2024_h1 = forecast[['ds', 'yhat']].tail(1)['yhat'].values[0]  # 2024 상반기 예측값

    # 3. 원본 데이터에 mean_cluster_prophet 컬럼 추가
    group_data['mean_cluster_prophet'] = None  # 빈 컬럼 생성

    # 기존 반기별 평균을 mean_cluster_prophet에 할당
    for i, row in semi_annual_data.iterrows():
        mask = (group_data['year'] == row['ds'].year) & (group_data['half_year'] == ('H1' if row['ds'].month == 6 else 'H2'))
        group_data.loc[mask, 'mean_cluster_prophet'] = row['y']

    # 2024년 상반기에는 예측값을 넣어줌
    group_data.loc[(group_data['year'] == 2024) & (group_data['half_year'] == 'H1'), 'mean_cluster_prophet'] = predicted_2024_h1

    # 클러스터별 데이터를 최종 결과에 저장
    final_results = pd.concat([final_results, group_data], ignore_index=True)

# 최종 데이터를 파일로 저장
final_results.to_csv('cluster_mean_prophet_results.csv', index=False)

print("클러스터별 예측 및 반기 평균 값이 'cluster_mean_prophet_results.csv' 파일로 저장되었습니다.")

import pandas as pd
from geopy.distance import geodesic

data = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/data/subwayInfo.csv')

# 1km 이내에 있는 지하철역 개수를 세는 함수
def count_nearby_stations(row, data, radius=0.1):
    count = 0
    for idx, station in data.iterrows():
        if row.name != idx:  # 자기 자신은 제외
            distance = geodesic((row['latitude'], row['longitude']), (station['latitude'], station['longitude'])).km
            if distance <= radius:
                count += 1
    return count

# 각 지하철역에 대해 1km 이내에 있는 지하철역 개수 계산
data['nearby_stations'] = data.apply(count_nearby_stations, axis=1, data=data)

# 결과 확인
print(data)

import pandas as pd
from geopy.distance import geodesic


# 각 행마다 가장 가까운 교통 허브의 거리를 계산하는 함수
def find_nearest_hub(row, sub_4):
    min_distance = float('inf')
    for idx, hub in sub_4.iterrows():
        distance = geodesic((row['latitude'], row['longitude']), (hub['latitude'], hub['longitude'])).km
        if distance < min_distance:
            min_distance = distance
    return min_distance

# 각 행마다 nearest_transportation_hub 계산
data['nearest_transportation_hub'] = data.apply(find_nearest_hub, axis=1, sub_4=sub_4)

# 결과 확인
print(data)

import pandas as pd
from geopy.distance import geodesic

parkdata = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/data/newparkInfo.csv')
complexdata = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/EDA&FE/max_train_updated_clean.csv')
complexdata = complexdata.drop_duplicates(subset=['complex_id', 'latitude', 'longitude'])
complexdata = complexdata[['complex_id', 'latitude', 'longitude']]

# Parkdata에서 area가 10만 이상인 값 필터링
filtered_parkdata = parkdata[parkdata['area'] >= 100000]

# 각 공원 주변 1.5km 이내의 complexdata 개수를 세는 함수
def count_nearby_complexes(row, complexdata, radius=1.5):
    count = 0
    for idx, complex_row in complexdata.iterrows():
        distance = geodesic((row['latitude'], row['longitude']), (complex_row['latitude'], complex_row['longitude'])).km
        if distance <= radius:
            count += 1
    return count

# 각 공원에 대해 1.5km 이내에 있는 complexdata 개수 계산
filtered_parkdata['nearby_complexes'] = filtered_parkdata.apply(count_nearby_complexes, axis=1, complexdata=complexdata)

# 1.5km 이내에 complexdata가 10개 이상인 공원 필터링
result = filtered_parkdata[filtered_parkdata['nearby_complexes'] >= 10]

# 결과 확인
print(result)

import pandas as pd

data = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/data/fe_final_2.csv')

data = data.groupby('contract_year_month', as_index=False).agg({
    'deposit' : 'mean',
    'area_m2' : 'mean',
    'floor' : 'mean',
    'previous_month_interest_rate' : 'first',
    'age' : 'mean',
    'contract_count_last_month' : 'mean',
})

# 기본 변수
data['year'] = data['contract_year_month'].astype(str).str[:4].astype(int)
data['month'] = data['contract_year_month'].astype(str).str[4:6].astype(int)

# 전월 대비 변화율 변수 생성
data['area_m2_diff'] = data['area_m2'].diff()
data['floor_diff'] = data['floor'].diff()
data['interest_rate_diff'] = data['previous_month_interest_rate'].diff()
data['age_diff'] = data['age'].diff()
data['contract_count_last_month_diff'] = data['contract_count_last_month'].diff()

# 이동 평균 변수 생성
data['area_m2_ma'] = data['area_m2'].rolling(window=3).mean()
data['floor_ma'] = data['floor'].rolling(window=3).mean()
data['interest_rate_ma'] = data['previous_month_interest_rate'].rolling(window=3).mean()
data['age_ma'] = data['age'].rolling(window=3).mean()
data['contract_count_last_month_ma'] = data['contract_count_last_month'].rolling(window=3).mean()
data['time_index'] = range(len(data))


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 독립 변수와 종속 변수 설정
# 여기서 독립 변수로 사용할 컬럼들 선택 (예시로 일부 사용)
# Step 1: 데이터 분할
train_data = data[data['contract_year_month'] <= 202312]  # 202312 이전 데이터
test_data = data[(data['contract_year_month'] >= 202401) & (data['contract_year_month'] <= 202406)]  # 202401 ~ 202406 데이터

# 학습을 위한 독립 변수와 종속 변수 설정
X_train = train_data.drop(['deposit'], axis=1)
y_train = train_data['deposit']

# 테스트 데이터를 위한 독립 변수 설정
X_test = test_data.drop(columns=['deposit'])





# 선형회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 회귀 계수 확인
coefficients = model.coef_
intercept = model.intercept_

# 인덱스 생성
# 회귀 계수를 X 데이터에 곱하고 가중합 계산
train_data['deposit_index'] = np.dot(X_train, coefficients) + intercept

import pandas as pd


# 증감 방향 계산
# 현재와 이전 값의 차이를 계산
train_data['deposit_change'] = train_data['deposit'].diff().fillna(0)
train_data['deposit_index_change'] = train_data['deposit_index'].diff().fillna(0)

# 증감 방향 비교
train_data['same_direction'] = np.where(
    (train_data['deposit_change'] > 0) & (train_data['deposit_index_change'] > 0) | 
    (train_data['deposit_change'] < 0) & (train_data['deposit_index_change'] < 0),
    1,  # 같은 방향
    0   # 다른 방향
)

# 비율 계산
same_direction_ratio = train_data['same_direction'].mean()

# 결과 출력
print(f"증감 방향이 같은 비율: {same_direction_ratio:.2f}")

# Step 3: 202401 ~ 202406 데이터 선택
forecast_data = data[(data['contract_year_month'] >= 202401) & (data['contract_year_month'] <= 202406)]

# 독립 변수 준비
X_forecast = forecast_data.drop(['deposit'], axis=1)

# Step 4: 예측 수행
forecast_data['deposit_index'] = np.dot(X_test, model.coef_) + model.intercept_

train_data['deposit_change'] = train_data['deposit'].diff().fillna(0)
train_data['deposit_index_change'] = train_data['deposit_index'].diff().fillna(0)
forecast_data['deposit_index_change'] = forecast_data['deposit_index'].diff().fillna(0)

data = pd.concat([train_data, forecast_data])

data2 = data[['contract_year_month','deposit_index', 'deposit_index_change']]

data2.to_csv('deposit_index.csv', index=False)

import pandas as pd
from scipy.spatial import cKDTree

# 샘플 데이터 로딩 (경도, 위도 정보가 포함된 데이터프레임)
data = pd.read_csv('your_data.csv')

# K-D Tree 생성
tree = cKDTree(data[['latitude', 'longitude']].values)

# 각 지점에 대해 반경 1km 이내의 개수 계산
def count_nearby_stations(row, tree, radius=1):
    # 위도, 경도를 쿼리하여 반경 내 지점 찾기 (1km = 약 0.009 degrees)
    indices = tree.query_ball_point([row['latitude'], row['longitude']], r=radius / 111)
    return len(indices) - 1  # 자기 자신 제외

data['near_complex_num'] = data.apply(lambda row: count_nearby_stations(row, tree), axis=1)

print(data)


import numpy as np

# 파일 불러오기
interest_rate_df = pd.read_csv(file_path + 'interestRate.csv')

# index 기준으로 역순 정렬
interest_rate_df = interest_rate_df.sort_index(ascending=False).reset_index(drop=True)

# 기울기를 계산할 함수 정의
def calculate_slope(series):
    if len(series) < 3:
        return np.nan  # 데이터가 3개월 미만일 때는 NaN 처리
    x = np.array(range(len(series)))
    y = np.array(series)
    slope = np.polyfit(x, y, 1)[0]  # 기울기 계산
    return slope

# 3개월치의 기울기를 계산해서 새로운 열로 추가
interest_rate_df['3_month_slope'] = interest_rate_df['interest_rate'].rolling(window=3).apply(calculate_slope, raw=True)

# 첫 번째 NaN 값 대체
first_valid_slope = interest_rate_df['3_month_slope'].dropna().iloc[0]
interest_rate_df['3_month_slope'].fillna(first_valid_slope, inplace=True)

# index 0은 0으로 대체
interest_rate_df.loc[0, '3_month_slope'] = 0

# index 1은 2개월치의 기울기로 계산
if len(interest_rate_df) > 1:
    x = np.array([0, 1])
    y = interest_rate_df.loc[0:1, 'interest_rate']
    two_month_slope = np.polyfit(x, y, 1)[0]
    interest_rate_df.loc[1, '3_month_slope'] = two_month_slope

# df와 interest_rate_df를 'contract_year_month'를 기준으로 병합
df = pd.merge(df, interest_rate_df[['contract_year_month', '3_month_slope']], 
              on='contract_year_month', 
              how='left')

# 결과 확인
print(df.head())


import pandas as pd
import numpy as np

data = pd.read_csv('/data/ephemeral/home/level2-competitiveds-recsys-05/notebooks/code copy/final2.csv')

# 변수 목록
variables = ['area_m2', 'floor', 'built_year', 
             'max_deposit_per_area', 'previous_deposit2', 'nearest_transportation_hub']

# 카테고리 수
num_categories = 6

# 각 변수에 대해 min-max 스케일링과 카테고리화
for var in variables:
    # Min-max 스케일링
    scaled_values = (data[var] - data[var].min()) / (data[var].max() - data[var].min())
    
    # 카테고리화
    data[f'{var}_categorical'] = pd.cut(scaled_values, bins=num_categories, labels=[i+1 for i in range(num_categories)])

# 결과 확인
print(data.head())

# 1. 데이터 정렬
df = df.sort_values(by=['complex_id', 'area_m2', 'contract_year_month', 'contract_day']).reset_index(drop=True)

# 2. 월별 deposit 평균 계산
monthly_avg_deposit = df.groupby('contract_year_month')['deposit'].mean().to_dict()

# 3. 이전 deposit 가져오기: shift
df['previous_deposit'] = df.groupby(['complex_id', 'area_m2'])['deposit'].shift(1)

# 4. 첫 번째 거래에서 previous_deposit가 NaN이고, deposit이 존재하는 경우 처리
df['transaction_order'] = df.groupby(['complex_id', 'area_m2']).cumcount()
mask = (df['previous_deposit'].isna()) & (df['transaction_order'] == 0) & (df['deposit'].notna())
df.loc[mask, 'previous_deposit'] = df.loc[mask, 'deposit']

# 5. 증감 비율 적용하여 previous_deposit 보정
def adjust_deposit_with_ratio(row):
    if pd.isna(row['previous_deposit']):
        return np.nan # 결측이면 패스

    prev_month = row['contract_year_month_prev']
    curr_month = row['contract_year_month']

    # 월별 deposit 평균 가져오기
    if prev_month in monthly_avg_deposit and curr_month in monthly_avg_deposit:
        ratio = monthly_avg_deposit[curr_month] / monthly_avg_deposit[prev_month]
    else:
        ratio = 1 # 평균이 없을 경우 비율을 1로 설정

    # previous_deposit 비율 적용
    return row['previous_deposit'] * ratio

df['contract_year_month_prev'] = df.groupby(['complex_id', 'area_m2'])['contract_year_month'].shift(1)

# 6. 보정된 previous_deposit 적용
df['previous_deposit2'] = df.apply(adjust_deposit_with_ratio, axis=1)

# 8. remaining NaN 값 처리 (기존 로직 유지)
valid_deposits = df[df['deposit'].notna()]
last_valid_deposits = valid_deposits.groupby(['complex_id', 'area_m2'])['deposit'].last().reset_index()

complex_id_to_area_deposit = {}
for complex_id, group in last_valid_deposits.groupby('complex_id'):
    area_m2_values = group['area_m2'].values
    deposit_values = group['deposit'].values
    complex_id_to_area_deposit[complex_id] = (area_m2_values, deposit_values)

missing_prev_deposit_idx = df[df['previous_deposit'].isna()].index
missing_prev_deposit_rows = df.loc[missing_prev_deposit_idx]

for complex_id, group in missing_prev_deposit_rows.groupby('complex_id'):
    if complex_id in complex_id_to_area_deposit:
        valid_area_m2_values, valid_deposit_values = complex_id_to_area_deposit[complex_id]
        missing_area_m2_values = group['area_m2'].values

        # numpy를 사용하여 가장 가까운 area_m2 찾기
        diff = np.abs(missing_area_m2_values[:, np.newaxis] - valid_area_m2_values[np.newaxis, :])
        min_idx = diff.argmin(axis=1)

        # 해당 deposit 값 가져오기
        deposit_values_to_fill = valid_deposit_values[min_idx]

        # previous_deposit 컬럼 업데이트
        df.loc[group.index, 'previous_deposit'] = deposit_values_to_fill

# 8. remaining NaN 값 처리 (기존 로직 유지)
valid_deposits = df[df['deposit'].notna()]
last_valid_deposits = valid_deposits.groupby(['complex_id', 'area_m2'])['deposit'].last().reset_index()

complex_id_to_area_deposit = {}
for complex_id, group in last_valid_deposits.groupby('complex_id'):
    area_m2_values = group['area_m2'].values
    deposit_values = group['deposit'].values
    complex_id_to_area_deposit[complex_id] = (area_m2_values, deposit_values)

missing_prev_deposit_idx = df[df['previous_deposit2'].isna()].index
missing_prev_deposit_rows = df.loc[missing_prev_deposit_idx]

for complex_id, group in missing_prev_deposit_rows.groupby('complex_id'):
    if complex_id in complex_id_to_area_deposit:
        valid_area_m2_values, valid_deposit_values = complex_id_to_area_deposit[complex_id]
        missing_area_m2_values = group['area_m2'].values

        # numpy를 사용하여 가장 가까운 area_m2 찾기
        diff = np.abs(missing_area_m2_values[:, np.newaxis] - valid_area_m2_values[np.newaxis, :])
        min_idx = diff.argmin(axis=1)

        # 해당 deposit 값 가져오기
        deposit_values_to_fill = valid_deposit_values[min_idx]

        # previous_deposit 컬럼 업데이트
        df.loc[group.index, 'previous_deposit2'] = deposit_values_to_fill
