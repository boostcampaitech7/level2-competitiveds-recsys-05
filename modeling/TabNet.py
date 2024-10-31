#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[4]:


import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 파일 경로 설정
file_path = '../data/'

# 파일 불러오기
df = pd.read_csv(file_path + '123.csv')
sample_submission = pd.read_csv(file_path + 'sample_submission.csv')


# In[5]:


# train, test split
train = df[df["_type"] == "train"]
test = df[df["_type"] == "test"]


# In[12]:


from sklearn.model_selection import train_test_split

holdout_start = 202307
holdout_end = 202312
holdout_data = train[(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]
train_data = train[~(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]

# drop 수정 필요
X_train_full = train_data.drop(['deposit', '_type'], axis=1)
y_train_full = train_data['deposit']
X_holdout = holdout_data.drop(['deposit', '_type'], axis=1)
y_holdout = holdout_data['deposit']
X_test = test.drop(['deposit', '_type'], axis=1)

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# # 모델링

# In[ ]:


from pytorch_tabnet.tab_model import TabNetRegressor
import torch
# TabNet 모델 정의
tabNet = TabNetRegressor(
    n_d=8,  # TabNet의 Decision Transformer의 출력 차원
    n_a=8,  # Attention Mechanism의 출력 차원
    n_steps=3,  # TabNet에서 step 수
    gamma=1.5,  # regularization term
    lambda_sparse=0.001,  # sparse regularization strength
    optimizer_fn=torch.optim.Adam,  # Optimizer로 Adam 사용
    optimizer_params=dict(lr=1e-2),  # Learning rate 설정
)

# 모델 학습
history = tabNet.fit(
    X_train.to_numpy(),
    y_train.values.reshape(-1, 1),
    eval_set=[(X_val.to_numpy(), y_val.values.reshape(-1, 1))],
    max_epochs=10,
    eval_metric=['mae'],
    patience=50,  # early stopping patience
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=2
)


# # 하이퍼 파라미터 튜닝

# # holdout 검증

# In[ ]:


holdout_pred = tabNet.predict(X_holdout)
holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
holdout_rmse = root_mean_squared_error(y_holdout, holdout_pred)

print("Holdout 데이터셋 성능")
print(f"LightGBM MAE: {holdout_mae:.2f}")
print(f"LightGBM RMSE: {holdout_rmse:.2f}")


# # 재학습 후 output 생성 (제출용)

# In[ ]:


# train, test split
train_data = df[df["_type"] == "train"]
test_data = df[df["_type"] == "test"]

X_train = train_data[columns]
y_train = train_data['deposit']
X_test = test_data[columns]


# In[ ]:


# 재학습

history2 = tabNet2.fit(
    X_train.to_numpy(),
    y_train.values.reshape(-1, 1),
    eval_set=[(X_val.to_numpy(), y_val.values.reshape(-1, 1))],
    max_epochs=10,
    eval_metric=['mae'],
    patience=50,  # early stopping patience
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=2
)


# In[ ]:


# 제출용 csv 생성
sample_submission["deposit"] = y_test_pred
sample_submission.to_csv("output2.csv", index= False)

