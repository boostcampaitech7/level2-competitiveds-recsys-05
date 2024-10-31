#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 파일 경로 설정
file_path = '../data/'

# 파일 불러오기
df = pd.read_csv(file_path + 'final.csv')
sample_submission = pd.read_csv(file_path + 'sample_submission.csv')


# In[2]:


# train, test split
train = df[df["_type"] == "train"].sort_values(by='index')
test = df[df["_type"] == "test"].sort_values(by='index')


# In[ ]:


columns = ['area_m2', 'contract_type', 'floor', 'latitude', 'longitude', 'age',
       'complex_id', 'max_deposit', 'cluster_labels',
       'contract_year', 'contract_month', 'mean_deposit_per_area_year',
       'max_deposit_per_area', 'previous_deposit',
       'half_max_deposit', 'deposit_std_id', 'pred_deposit',
       'deposit_label', 'nearest_subway_distance_km',
       'nearest_elementary_distance_km', 'nearest_middle_distance_km',
       'nearest_high_distance_km', 'nearest_park_distance_km',
       'nearest_park_area', 'num_subway_within_0_5', 'num_subway_within_1',
       'num_subway_within_3', 'num_elementary_within_0_5',
       'num_elementary_within_1', 'num_elementary_within_2',
       'num_middle_within_0_5', 'num_middle_within_1', 'num_middle_within_2',
       'num_high_within_0_5', 'num_high_within_1', 'num_high_within_2',
       'num_park_within_0_8', 'num_park_within_1_5', 'num_park_within_2',
       'area_floor_interaction']


# In[3]:


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

# In[4]:


def objective(trial):
    params = {
        'device': trial.suggest_categorical('device', ['cuda']),
        'objective': trial.suggest_categorical('objective', ['reg:absoluteerror']),
        'booster': trial.suggest_categorical('booster',['gbtree']),
        'tree_method': trial.suggest_categorical('tree_method',['hist']),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.01, 0.1, 0.2]),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_leaves': trial.suggest_int('max_leaves', 0, 255),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': trial.suggest_categorical('eval_metric',['mae']),
        'random_state': trial.suggest_categorical('random_state', [RANDOM_SEED])
    }

    # DMatrix 객체로 변환
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dholdout = xgb.DMatrix(X_holdout, label=y_holdout)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int('num_boost_round', 50, 500),
        evals=[(dtrain, 'train'), (dval, 'val')],
        callbacks=[optuna.integration.XGBoostPruningCallback(trial, 'val-mae')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    holdout_pred = model.predict(dholdout)
    
    holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
    holdout_rmse = root_mean_squared_error(y_holdout, holdout_pred)

    trial.set_user_attr("rmse", holdout_rmse)
    
    return holdout_mae


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


# # holdout 검증

# In[5]:


print("Best trial:")
trial = study.best_trial

print(f"MAE: {trial.value}")
print(f"RMSE: {trial.user_attrs['rmse']}")
print("Best hyperparameters: ", trial.params)


# # 재학습 후 output 생성 (제출용)

# In[6]:


# 재학습
best_params = trial.params
best_model = xgb.XGBRegressor(**best_params)

best_model.fit(train[columns], train['deposit'])

y_pred = best_model.predict(train[columns])
y_test_pred = best_model.predict(test[columns])

mae = mean_absolute_error(train['deposit'], y_pred)
rmse = root_mean_squared_error(train['deposit'], y_pred)

print(f" MAE: {mae:.2f}")
print(f" RMSE: {rmse:.2f}")


# In[7]:


# 제출용 csv 생성
sample_submission["deposit"] = y_test_pred
sample_submission.to_csv("output.csv", index= False)

