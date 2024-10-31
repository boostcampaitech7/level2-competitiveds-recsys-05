#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 파일 경로 설정
file_path = '../data/'

# 파일 불러오기
df = pd.read_csv(file_path + 'final.csv')
sample_submission = pd.read_csv(file_path + 'sample_submission.csv')


# In[3]:


# train, test split
train = df[df["_type"] == "train"].sort_values(by='index')
test = df[df["_type"] == "test"].sort_values(by='index')


# In[4]:


columns = [
    'area_m2',
    'contract_type', 'floor', 'age',
    'previous_month_interest_rate',
    'monthly_new_supply', 'complex_id', 'max_deposit', 'contract_year',
    'contract_month', 'mean_cluster_prophet',
    'max_deposit_per_area', 'previous_deposit',
    'half_max_deposit', 'deposit_std_id', 
    'nearest_subway_distance_km', 'nearest_elementary_distance_km',
    'nearest_middle_distance_km', 'nearest_high_distance_km',
    'nearest_park_distance_km', 'nearest_park_area',
    'num_subway_within_0_5', 'num_subway_within_1', 'num_subway_within_3',
    'num_elementary_within_0_5', 'num_elementary_within_1',
    'num_elementary_within_2', 'num_middle_within_0_5',
    'num_middle_within_1', 'num_middle_within_2', 'num_high_within_0_5',
    'num_high_within_1', 'num_high_within_2', 'num_park_within_0_8',
    'num_park_within_1_5', 'num_park_within_2', 'area_floor_interaction',
    'pred_deposit', 'deposit_label'
]


# In[7]:


holdout_start = 202307
holdout_end = 202312
holdout_data = train[(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]
train_data = train[~(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]

X_train_full = train_data[columns]
y_train_full = train_data['deposit']
X_holdout = holdout_data[columns]
y_holdout = holdout_data['deposit']
X_test = test[columns]

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# # 모델링

# In[34]:


def objective(trial):
    params = {
    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
    'learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.01, 0.1, 0.2]),
    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    'max_depth': trial.suggest_int('max_depth', 1, 16),
    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'objective': trial.suggest_categorical('objective', ['regression_l1']),
    'random_state': trial.suggest_categorical('random_state', [42]),
    'verbose': trial.suggest_categorical('verbose', [-1])
    }
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'l1'), lgb.early_stopping(50, verbose=False)]
    )
    
    holdout_pred = model.predict(X_holdout)
    
    holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
    holdout_rmse = root_mean_squared_error(y_holdout, holdout_pred)

    trial.set_user_attr("rmse", holdout_rmse)
    
    return holdout_mae
  

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


# # holdout 검증

# In[35]:


print("Best trial:")
trial = study.best_trial

print(f"MAE: {trial.value}")
print(f"RMSE: {trial.user_attrs['rmse']}")
print("Best hyperparameters: ", trial.params)


# # 재학습 후 output 생성 (제출용)

# In[36]:


# 재학습
best_params = trial.params
best_model = lgb.LGBMRegressor(**best_params)

best_model.fit(train[columns], train['deposit'])

y_pred = best_model.predict(train[columns])
y_test_pred = best_model.predict(test[columns])

mae = mean_absolute_error(train['deposit'], y_pred)
rmse = root_mean_squared_error(train['deposit'], y_pred)

print(f" MAE: {mae:.2f}")
print(f" RMSE: {rmse:.2f}")


# In[37]:


# 제출용 csv 생성
sample_submission["deposit"] = y_test_pred
sample_submission.to_csv("output.csv", index= False)

