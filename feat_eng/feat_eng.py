import pandas as pd
import numpy as np

# 위도, 경도를 기준으로 아파트 단지 ID 만들기
def create_complex_id(df, lat_col='latitude', lon_col='longitude'):
    """
    위도와 경도를 기준으로 아파트 단지 ID를 생성하고, 원본 데이터프레임에 추가합니다.
    
    Parameters:
    df (pandas.DataFrame): 원본 데이터프레임
    lat_col (str): 위도 컬럼 이름 (기본값: 'latitude')
    lon_col (str): 경도 컬럼 이름 (기본값: 'longitude')
    
    Returns:
    pandas.DataFrame: 'complex_id'가 추가된 데이터프레임
    """
    # 고유 아파트 단지 개수 구하기 (latitude, longitude의 조합)
    unique_complexes = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)

    # 고유한 ID 할당
    unique_complexes['complex_id'] = unique_complexes.index

    # 원본 df에 'complex_id' 추가
    df_with_id = df.merge(unique_complexes, on=[lat_col, lon_col], how='left')
    
    return df_with_id

# Distance to the Nearest POI (km)
def nearest_POI(apt, POI):
    from sklearn.neighbors import NearestNeighbors
    
    apt_coord = apt[['latitude', 'longitude']].values
    POI_coord = POI[['latitude', 'longitude']].values

    # 각 좌표를 라디안으로 변환
    apt_coord_rad = np.radians(apt_coord)
    POI_coord_rad = np.radians(POI_coord)

    # Haversine 공식을 이용하여 NearestNeighbors 모델 사용:
    nbrs_POI = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine')
    nbrs_POI.fit(POI_coord_rad)

    # 가장 가까운 POI까지의 거리 계산
    distance_POI, indices = nbrs_POI.kneighbors(apt_coord_rad)
    # 거리를 km으로 변환
    distance_POI_km = distance_POI * 6371 # 지구 반지름(km)

    return distance_POI_km[:, 0], indices[:, 0]

# BallTree를 사용한 반경 내 시설물 개수 계산
def count_within_radius(apt, POI, radius_km):
    from sklearn.neighbors import BallTree
    earth_raidus = 6371.0 # 지구 반지름(km)
    radius = radius_km / earth_raidus # 반경을 라디안으로 변경
    
    # 아파트 좌표
    apt_coord = np.radians(apt[['latitude', 'longitude']].values)
    
    # POI 좌표
    POI_coord = np.radians(POI[['latitude', 'longitude']].values)
    
    tree = BallTree(POI_coord, metric='haversine')
    counts = tree.query_radius(apt_coord, r=radius, count_only=True)
    return counts

def add_interest_rate(interest_df, diff=False, roll=False, window=None):
    """
    이전 달 금리 변수 추가 및 차분, 롤링 금리 계산
    주의: interestRate.csv는 계약 연월 기준 내림차순으로 정렬되어 있음

    Args:
        interest_df (float): 각 연월에 해당하는 금리
        diff (bool, optional): 금리의 차분 값 계산 여부. Defaults to False.
        roll (bool, optional): 금리에 대해 롤링 평균을 할지 여부. Defaults to False.
        window (int, optional): 롤링 평균을 계산할 윈도우 크기. Defaults to None.

    Returns:
        DataFrame: 이전달 금리 (와 금리의 롤링 평균) 변수를 추가한 데이터프레임
    """
    
    # 계약 연월 기준 오름차순 정렬
    interest_df = interest_df.sort_values(by='contract_year_month', ascending=True).reset_index(drop=True)
    
    # 이전 달 금리 추가
    interest_df['previous_month_interest_rate'] = interest_df['interest_rate'].shift(1).values
    
    # 202406에 해당하는 값 추가 (마지막 행에 추가)
    new_interest = pd.DataFrame({'contract_year_month': [202406], 
                                 'interest_rate': [np.nan], 
                                 'previous_month_interest_rate': interest_df[interest_df['contract_year_month']==202405]['interest_rate'].values})
    
    interest_df = pd.concat([interest_df, new_interest], ignore_index=True)
    
    # 차분 계산 (diff가 True일 경우)
    if diff:
        interest_df['previous_month_interest_rate_diff'] = interest_df['previous_month_interest_rate'].diff().bfill()

    # 롤링 평균 계산 (roll이 True일 경우)
    if roll:
        interest_df[f'previous_{window}month_interest_rate_roll'] = interest_df['previous_month_interest_rate'].rolling(window=window, min_periods=1).mean()
    
    return interest_df

# 신축 공급량과 비슷한 효과를 내도록 하는 age 관련 변수
def add_new_supply_column(df, age_col='age', group_col='contract_year_month', supply_col_name='monthly_new_supply'):
    """
    주어진 데이터프레임에서 age가 0인 값들을 contract_year_month 기준으로 계산하여
    new_supply 컬럼을 추가하는 함수.

    Parameters:
    - df: 입력 데이터프레임
    - age_col: age가 기록된 컬럼 이름 (기본값은 'age')
    - group_col: 그룹화 컬럼 이름 (기본값은 'contract_year_month')

    Returns:
    - new_supply: group_col별 공급물량 numpy array
    """
    # age 값이 0인 행들만 선택
    age_zero = df[df[age_col] == 0]

    # contract_year_month별로 age가 0인 데이터의 개수를 세어 new_supply로 집계
    new_supply = age_zero.groupby(group_col).size().reset_index(name=supply_col_name)

    return new_supply

# 그룹별 기간별 이전 시점의 계약 건수
def add_previous_contract_count(df, group_col = 'Cluster', grouping_type='year'):
    data = df.copy()
    data['contract_year_month'] = data['contract_year_month'].astype(str)
    data['contract_year'] = data['contract_year_month'].str[:4].astype(int)
    data['contract_month'] = data['contract_year_month'].str[4:6].astype(int)

    if grouping_type == 'year':
        # 이전 년도 계약 건수 계산
        contract_count = data.groupby([group_col, 'contract_year']).size().reset_index(name='contract_count_last_year')
        contract_count['contract_year'] += 1
        data = data.merge(contract_count, on=[group_col, 'contract_year'], how='left')
        return data['contract_count_last_year'].fillna(0).astype(int)

    elif grouping_type == 'month':
        # 이전 달 계약 건수 계산
        contract_count_last_month = data.groupby(['Cluster', 'contract_year_month']).size().reset_index(name='contract_count_last_month')
        contract_count_last_month['contract_year'] = contract_count_last_month['contract_year_month'].str[:4].astype(int)
        contract_count_last_month['contract_month'] = contract_count_last_month['contract_year_month'].str[4:6].astype(int)
        
        contract_count_last_month['contract_month'] = contract_count_last_month['contract_month'] + 1
        contract_count_last_month.loc[contract_count_last_month['contract_month']==13, 'contract_year'] = contract_count_last_month['contract_year'] + 1
        contract_count_last_month.loc[contract_count_last_month['contract_month']==13, 'contract_month'] = 1

        contract_count_last_month['contract_year_month'] = contract_count_last_month['contract_year'].astype(str).str.zfill(4) + contract_count_last_month['contract_month'].astype(str).str.zfill(2)
        contract_count_last_month = contract_count_last_month.drop(['contract_year', 'contract_month'], axis=1)

        data = data.merge(contract_count_last_month, on=['Cluster', 'contract_year_month'], how='left')
        return data['contract_count_last_month'].fillna(0).astype(int)

    elif grouping_type == 'quarter':
        # 이전 분기 계약 건수 계산
        # 분기 계산 (1분기: 1~3월, 2분기: 4~6월, 3분기: 7~9월, 4분기: 10~12월)
        data['contract_quarter'] = ((data['contract_month'] - 1) // 3) + 1
        data['contract_year_quarter'] = data['contract_year'].astype(str) + 'Q' + data['contract_quarter'].astype(str)

        contract_count_last_quarter = data.groupby(['Cluster', 'contract_year_quarter']).size().reset_index(name='contract_count_last_quarter')
        contract_count_last_quarter['contract_year'] = contract_count_last_quarter['contract_year_quarter'].str[:4].astype(int)
        contract_count_last_quarter['contract_quarter'] = contract_count_last_quarter['contract_year_quarter'].str[5:6].astype(int)

        contract_count_last_quarter['contract_quarter'] = contract_count_last_quarter['contract_quarter'] + 1
        contract_count_last_quarter.loc[contract_count_last_quarter['contract_quarter']==5, 'contract_year'] = contract_count_last_quarter['contract_year'] + 1
        contract_count_last_quarter.loc[contract_count_last_quarter['contract_quarter']==5, 'contract_quarter'] = 1

        contract_count_last_quarter['contract_year_quarter'] = contract_count_last_quarter['contract_year'].astype(str) + 'Q' + contract_count_last_quarter['contract_quarter'].astype(str)
        contract_count_last_quarter = contract_count_last_quarter.drop(['contract_year', 'contract_quarter'], axis=1)

        data = data.merge(contract_count_last_quarter, on=['Cluster', 'contract_year_quarter'], how='left')
        return data['contract_count_last_quarter'].fillna(0).astype(int)

    elif grouping_type == 'half':
        # 이전 상/하반기 계약 건수 계산
        # 상/하반기 계산 (상반기: 1~6월, 하반기: 7~12월)
        data['contract_half'] = data['contract_month'].apply(lambda x: 1 if x <= 6 else 2)
        data['contract_year_half'] = data['contract_year'].astype(str) + 'H' + data['contract_half'].astype(str)

        contract_count_last_half = data.groupby(['Cluster', 'contract_year_half']).size().reset_index(name='contract_count_last_half')
        contract_count_last_half['contract_year'] = contract_count_last_half['contract_year_half'].str[:4].astype(int)
        contract_count_last_half['contract_half'] = contract_count_last_half['contract_year_half'].str[5:6].astype(int)

        contract_count_last_half['contract_half'] = contract_count_last_half['contract_half'] + 1
        contract_count_last_half.loc[contract_count_last_half['contract_half']==3, 'contract_year'] = contract_count_last_half['contract_year'] + 1
        contract_count_last_half.loc[contract_count_last_half['contract_half']==3, 'contract_half'] = 1

        contract_count_last_half['contract_year_half'] = contract_count_last_half['contract_year'].astype(str) + 'H' + contract_count_last_half['contract_half'].astype(str)
        contract_count_last_half = contract_count_last_half.drop(['contract_year', 'contract_half'], axis=1)

        data = data.merge(contract_count_last_half, on=['Cluster', 'contract_year_half'], how='left')
        return data['contract_count_last_half'].fillna(0).astype(int)

    else:
        raise ValueError("grouping_type must be one of 'year', 'month', 'quarter', 'half'")
    
# 그룹별 기간별 전세가 평균 및 표준편차 추가
def calculate_deposit_stats(df, group_col='Cluster', time_unit='year'):
    data = df.copy()
    
    # 시간 단위에 따라 새로운 열 생성
    if time_unit == 'year':
        data['time_group'] = data['contract_year']
    elif time_unit == 'month':
        data['time_group'] = data['contract_year_month']
    elif time_unit == 'quarter':
        # 분기 계산 (1~4)
        data['quarter'] = ((data['contract_year_month'] % 100 - 1) // 3) + 1
        data['time_group'] = data['contract_year'].astype(str) + 'Q' + data['quarter'].astype(str)
    elif time_unit == 'half':
        # 상/하반기 계산
        data['half'] = ((data['contract_year_month'] % 100 - 1) // 6) + 1
        # 상반기: 1월, 하반기: 7월로 설정하여 'YYYYMM' 형태로 만듦
        data['month'] = data['half'].apply(lambda x: '01' if x == 1 else '07')
        data['time_group'] = data['contract_year'].astype(str) + data['month']
        data['time_group'] = data['time_group'].astype(int)
    else:
        raise ValueError("Invalid time_unit. Choose from 'year', 'month', 'quarter', 'half'.")
    
    # 'train'과 'test' 데이터 분리
    train_df = data[data['_type'] == 'train'].copy()
    test_df = data[data['_type'] == 'test'].copy()
    
    # 변수명 설정
    mean_col = f'deposit_mean_last_{time_unit}'
    std_col = f'deposit_std_last_{time_unit}'
    
    # 각 (group_col, time_group)별 deposit의 평균과 표준편차 계산
    deposit_stats = train_df.groupby([group_col, 'time_group'])['deposit'].agg(['mean', 'std']).reset_index()
    deposit_stats.rename(columns={'mean': mean_col, 'std': std_col}, inplace=True)
    deposit_stats[mean_col] = deposit_stats[mean_col].fillna(0)
    deposit_stats[std_col] = deposit_stats[std_col].fillna(0)
    
    # 다음 기간의 time_group 생성 (현재 기간 +1)
    if time_unit == 'year':
        deposit_stats['next_time_group'] = deposit_stats['time_group'] + 1
    elif time_unit == 'month':
        # 'YYYYMM' 형식의 숫자를 datetime으로 변환
        deposit_stats['time_group'] = pd.to_datetime(deposit_stats['time_group'].astype(str), format='%Y%m')
        deposit_stats['next_time_group'] = deposit_stats['time_group'] + pd.DateOffset(months=1)
        deposit_stats['next_time_group'] = deposit_stats['next_time_group'].dt.strftime('%Y%m').astype(int)
        deposit_stats['time_group'] = deposit_stats['time_group'].dt.strftime('%Y%m').astype(int)
    elif time_unit == 'quarter':
        # 분기를 처리하기 위해 'time_group'을 다시 생성
        deposit_stats['quarter'] = deposit_stats['time_group'].str.extract(r'Q(\d)').astype(int)
        deposit_stats['year'] = deposit_stats['time_group'].str[:4].astype(int)
        deposit_stats['next_quarter'] = deposit_stats['quarter'] + 1
        deposit_stats.loc[deposit_stats['next_quarter'] > 4, 'next_quarter'] = 1
        deposit_stats.loc[deposit_stats['next_quarter'] == 1, 'year'] += 1
        deposit_stats['next_time_group'] = deposit_stats['year'].astype(str) + 'Q' + deposit_stats['next_quarter'].astype(str)
        deposit_stats = deposit_stats.drop(['quarter', 'year', 'next_quarter'], axis=1)
    elif time_unit == 'half':
        # 'YYYYMM'을 datetime으로 변환
        deposit_stats['time_group'] = pd.to_datetime(deposit_stats['time_group'].astype(str), format='%Y%m')
        deposit_stats['next_time_group'] = deposit_stats['time_group'] + pd.DateOffset(months=6)
        deposit_stats['next_time_group'] = deposit_stats['next_time_group'].dt.strftime('%Y%m').astype(int)
        deposit_stats['time_group'] = deposit_stats['time_group'].dt.strftime('%Y%m').astype(int)
    
    # train_df에 이전 시점의 평균과 표준편차 병합
    train_df = train_df.merge(
        deposit_stats[[group_col, 'next_time_group', mean_col, std_col]].rename(columns={'next_time_group': 'time_group'}),
        on=[group_col, 'time_group'],
        how='left'
    )
    train_df[[mean_col, std_col]] = train_df[[mean_col, std_col]].fillna(0)
    
    # test_df에 최신의 평균과 표준편차 병합
    latest_stats = deposit_stats.sort_values('time_group').groupby(group_col).tail(1)
    test_df = test_df.merge(
        latest_stats[[group_col, mean_col, std_col]],
        on=group_col,
        how='left'
    )
    test_df[[mean_col, std_col]] = test_df[[mean_col, std_col]].fillna(0)
    
    # train_df와 test_df 결합
    result_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 필요에 따라 추가적인 열 삭제
    result_df = result_df.drop(['time_group'], axis=1)
    for col in ['quarter', 'half', 'month']:
        if col in result_df.columns:
            result_df = result_df.drop(col, axis=1)
    
    return result_df

# 층수 및 면적 범주화
# 층수 범주화 함수
def categorize_floor(df):
    """
    각 아파트 단지(complex_id) 내에서 층수를 저층, 중층, 고층으로 범주화합니다.
    각 단지별로 최대 층수를 기준으로 저층, 중층, 고층을 자동으로 결정합니다.
    """
    # 각 단지(complex_id) 내에서 최대 층수 구하기
    max_floor_per_complex = df.groupby('complex_id')['floor'].transform('max')
    df['max_floor_per_complex'] = max_floor_per_complex  # max_floor_per_complex를 데이터프레임에 추가

    # 층수 범주화 함수
    def categorize_by_max_floor(floor, max_floor):
        if floor <= max_floor * 0.3:
            return 'Low'
        elif floor <= max_floor * 0.7:
            return 'Medium'
        else:
            return 'High'

    # 각 row에 대해 층수를 범주화
    df['floor_category'] = df.apply(lambda row: categorize_by_max_floor(row['floor'], row['max_floor_per_complex']), axis=1)

    df.drop(columns=['max_floor_per_complex'], inplace=True)  # max_floor_per_complex 컬럼 삭제

    return df


def categorize_area(df):
    """
    area_m2 변수를 Very Small, Small, Medium, Large로 구분하여 area_category 변수를 생성합니다.
    면적을 평 기준으로 범주화합니다.
    Very Small (소형): 60㎡ 이하 (18평 이하)
    Small (중소형): 60㎡ ~ 85㎡ (18평 ~ 25평)
    Medium (중형): 85㎡ ~ 135㎡ (25평 ~ 40평)
    Large (대형): 135㎡ 이상 (40평 이상)
    """
    bins = [0, 60, 85, 135, np.inf]  # 제곱미터 기준 구간 설정
    labels = ['Very Small', 'Small', 'Medium', 'Large']
    df['area_category'] = pd.cut(df['area_m2'], bins=bins, labels=labels, right=False)
    return df