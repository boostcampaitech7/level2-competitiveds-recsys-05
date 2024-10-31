import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from haversine import haversine, Unit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gc


import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

file_path = './data/'
df = pd.read_csv(os.path.join(file_path, 'final.csv'))

# 파일 불러오기
interest_rate_df = pd.read_csv(file_path + 'interestRate.csv')

# index 기준으로 역순 정렬
interest_rate_df = interest_rate_df.sort_index(ascending=False).reset_index(drop=True)
interest_rate_df = pd.concat([interest_rate_df, pd.DataFrame({'year_month': [202406], 'interest_rate': [3.56]})])

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
interest_rate_df = interest_rate_df.rename({'year_month': 'contract_year_month'}, axis=1)

df = df.merge(interest_rate_df[['contract_year_month', '3_month_slope']], on='contract_year_month', how='left')

# train, test split
train = df[df["_type"] == "train"].sort_values(by='index')
test = df[df["_type"] == "test"].sort_values(by='index')

columns = [
    'area_m2', 'floor',
    'built_year', 'latitude', 'longitude',
    'previous_month_interest_rate', 'monthly_new_supply',
    'complex_id', 'max_deposit', 'cluster_labels',
    'contract_year', 'contract_month',
    'mean_deposit_per_area_year',
    'max_deposit_per_area',
    'previous_deposit2',
    'half_max_deposit',
    'deposit_std_id',
    'nearest_subway_distance_km', 'nearest_elementary_distance_km',
    'nearest_middle_distance_km', 'nearest_high_distance_km',
    'nearest_park_distance_km', 'nearest_park_area',
    'num_subway_within_0_5', 'num_subway_within_1', 'num_subway_within_2',
    'num_elementary_within_0_5', 'num_elementary_within_1',
    'num_elementary_within_2', 'num_middle_within_0_5',
    'num_middle_within_1', 'num_middle_within_2', 'num_high_within_0_5',
    'num_high_within_1', 'num_high_within_2', 'num_park_within_0_8',
    'num_park_within_1_5', 'num_park_within_2',
    'area_floor_interaction',
    'nearest_transportation_hub',
    'contract_type',
    '3_month_slope'
]

holdout_start = 202307
holdout_end = 202312
holdout_data = train[(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]
train_data = train[~((train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end))]

X_train_full = train_data[columns]
y_train_full = train_data['deposit']
X_holdout = holdout_data[columns]
y_holdout = holdout_data['deposit']
X_test = test[columns]

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=42
)

# 범주형 컬럼 정의
cat_cols = ['contract_type']

# LightGBM과 XGBoost용 데이터 (Label Encoding)
X_train_num = X_train_full.copy()
X_holdout_num = X_holdout.copy()
X_test_num = X_test.copy()

X_train_num[cat_cols] = X_train_num[cat_cols].astype(int)
X_holdout_num[cat_cols] = X_holdout_num[cat_cols].astype(int)
X_test_num[cat_cols] = X_test_num[cat_cols].astype(int)

X_train_cat = X_train_full.copy()
X_holdout_cat = X_holdout.copy()
X_test_cat = X_test.copy()

X_train_cat[cat_cols] = X_train_cat[cat_cols].astype(int)
X_holdout_cat[cat_cols] = X_holdout_cat[cat_cols].astype(int)
X_test_cat[cat_cols] = X_test_cat[cat_cols].astype(int)

# LightGBM 모델 정의
lgb_params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 500,
    'max_depth': 15,
    'num_leaves': 96,
    'subsample': 0.9104369435751337,
    'colsample_bytree': 0.5371328882033616,
    'objective': 'regression_l1',
    'random_state': 42,
    'verbose': -1
}
lgb_model = lgb.LGBMRegressor(**lgb_params)

# XGBoost 모델 정의
xgb_params = {
    "device": "cuda",
    'booster': 'gbtree',
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "eval_metric": "mae",
    "seed": 42,
    "learning_rate": 0.2,
    "max_depth": 8,
    'max_leaves': 234,
    "colsample_bytree": 0.8,
    "n_estimators": 228
}
xgb_model = xgb.XGBRegressor(**xgb_params)

# CatBoost 모델 정의
cat_model = CatBoostRegressor(
    boosting_type = "Plain",
    eval_metric = "MAE",
    iterations = 500,
    learning_rate = 0.05,
    l2_leaf_reg = 10,
    max_leaves = 255,
    grow_policy='Lossguide',
    random_seed = 42,
    task_type = "GPU",
    depth = 16
)

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# 기본 모델 리스트
base_models = [lgb_model, xgb_model, cat_model]

# 메타 모델 정의 (선형 회귀 사용)
meta_model = LinearRegression()

# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 훈련 데이터에 대한 메타 피처 생성
train_meta_features = np.zeros((X_train_num.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test_num.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    print(f"Generating meta features for model {i+1} ({model.__class__.__name__})")
    oof_pred = np.zeros(X_train_num.shape[0])

    for train_idx, val_idx in kf.split(X_train_num):
        X_train_fold = X_train_num.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_num.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]

        # 모델별로 학습 방법 다르게 처리
        if isinstance(model, CatBoostRegressor):
            # CatBoost는 cat_features를 지정해야 합니다.
            model.fit(X_train_fold, y_train_fold, cat_features=cat_cols, verbose=False)
        else:
            model.fit(X_train_fold, y_train_fold)

        # 검증 데이터에 대한 예측
        oof_pred[val_idx] = model.predict(X_val_fold)

    # 훈련 데이터에 대한 OOF 예측 저장
    train_meta_features[:, i] = oof_pred

    # 전체 훈련 데이터로 다시 학습
    if isinstance(model, CatBoostRegressor):
        model.fit(X_train_cat, y_train_full, cat_features=cat_cols, verbose=False)
        test_meta_features[:, i] = model.predict(X_test_cat)
    else:
        model.fit(X_train_num, y_train_full)
        test_meta_features[:, i] = model.predict(X_test_num)

# 메타 피처 데이터프레임 생성
train_meta_df = pd.DataFrame(train_meta_features, columns=['lgb_pred', 'xgb_pred', 'cat_pred'])
test_meta_df = pd.DataFrame(test_meta_features, columns=['lgb_pred', 'xgb_pred', 'cat_pred'])

# 메타 모델 학습
meta_model.fit(train_meta_df, y_train_full)

# 메타 모델을 사용한 최종 예측
meta_pred_holdout = meta_model.predict(train_meta_df)
meta_pred_test = meta_model.predict(test_meta_df)

# 성능 평가
meta_mae = mean_absolute_error(y_train_full, meta_pred_holdout)
meta_rmse = np.sqrt(mean_squared_error(y_train_full, meta_pred_holdout))

print(f"Stacking Ensemble MAE: {meta_mae:.4f}")
print(f"Stacking Ensemble RMSE: {meta_rmse:.4f}")

# 최종 예측 저장
test_stacking_ensemble_pred = meta_pred_test

sample_submission = pd.read_csv(os.path.join(file_path, 'sample_submission.csv'))
sample_submission['deposit'] = test_stacking_ensemble_pred

sample_submission.to_csv(os.path.join(file_path, 'output_stacking.csv'), index=False)
