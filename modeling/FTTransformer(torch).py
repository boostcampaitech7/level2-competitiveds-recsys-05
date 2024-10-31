#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[79]:


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


# In[80]:


# train, test split
train = df[df["_type"] == "train"]
test = df[df["_type"] == "test"]


# In[81]:


from sklearn.model_selection import train_test_split

holdout_start = 202307
holdout_end = 202312
holdout_data = train[(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]
train_data = train[~(train['contract_year_month'] >= holdout_start) & (train['contract_year_month'] <= holdout_end)]

X_train_full = train_data.drop('deposit', axis=1)
y_train_full = train_data['deposit']
X_holdout = holdout_data.drop('deposit', axis=1)
y_holdout = holdout_data['deposit']
X_test = test.copy()

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# # 모델링

# ## FTTransformer

# Tree 모델과는 달리 스케일링 필요   
# 독립 변수(연속) : 여존슨 스케일링   
# 종속 변수 : 로그 변환   
# 범주형 변수 : 라벨 인코딩   

# In[82]:


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[83]:


X_train.columns


# In[84]:


categorical_columns = [
    'contract_type', 'complex_id'
]
cat_cardinalities = [df['contract_type'].unique().shape[0], df['complex_id'].unique().shape[0]]

continuous_columns = [
    'area_m2', 'contract_year_month', 'floor', 'latitude', 'longitude', 'age',
    'max_deposit', 'pred_deposit_per_area', 'pred_deposit'
]
n_cont_features = len(continuous_columns)


# In[87]:


from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
pt = PowerTransformer(method='yeo-johnson')
pt = RobustScaler()

X_train_cont = torch.Tensor(pt.fit_transform(X_train[continuous_columns])).to(device)
X_val_cont = torch.Tensor(pt.transform(X_val[continuous_columns])).to(device)
X_holdout_cont = torch.Tensor(pt.transform(X_holdout[continuous_columns])).to(device)

X_train_cat = torch.Tensor(X_train[categorical_columns].values).long().to(device)
X_val_cat = torch.Tensor(X_val[categorical_columns].values).long().to(device)
X_holdout_cat = torch.Tensor(X_holdout[categorical_columns].values).long().to(device)

y_train_log = torch.Tensor(np.log1p(y_train).values).to(device)
y_val_log = torch.Tensor(np.log1p(y_val).values).to(device)
y_holdout_log = torch.Tensor(np.log1p(y_holdout).values).to(device)


# In[73]:


from torch.utils.data import DataLoader, TensorDataset
# TensorDataset과 DataLoader를 사용해 배치 단위로 데이터 로드
batch_size = 256  # 배치 사이즈 설정

train_dataset = TensorDataset(X_train_cont, X_train_cat, y_train_log)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_cont, X_val_cat, y_val_log)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

holdout_dataset = TensorDataset(X_holdout_cont, X_holdout_cat, y_holdout_log)
holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)


# In[88]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import FTTransformer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm


# In[76]:


d_out = 1 # 회귀니까 1

default_kwargs = FTTransformer.get_default_kwargs() # 기본 파라미터
model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **default_kwargs,
    linformer_kv_compression_ratio=0.2,           # <---
    linformer_kv_compression_sharing='headwise',  # <---
).to(device)
criterion = nn.L1Loss()
optimizer = model.make_default_optimizer()

# 조기 종료 설정
best_val_loss = float('inf')
patience = 5  # 조기 종료를 위한 허용 에포크 수
counter = 0


# In[ ]:


default_kwargs


# In[78]:


# 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss_epoch = 0  # 에포크별 손실을 누적할 변수
    
    # 배치 단위로 학습
    for batch_data_cont, batch_data_cat, batch_target in tqdm(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
        train_loss = criterion(predictions, batch_target)

        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        train_loss = criterion(torch.expm1(predictions), torch.expm1(batch_target))
        train_loss_epoch += train_loss.item()
    
    train_loss_epoch /= len(train_loader)  # 배치 평균 손실 계산
    
    # 검증
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for batch_data_cont, batch_data_cat, batch_target in tqdm(val_loader):
            val_predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
            val_loss = criterion(torch.expm1(val_predictions), torch.expm1(batch_target))
            val_loss_epoch += val_loss.item()
    
    val_loss_epoch /= len(val_loader)  # 배치 평균 손실 계산

    print(f'Epoch {epoch+1}/{num_epochs}, Train MAE: {train_loss_epoch:.4f}, Val MAE: {val_loss_epoch:.4f}')

    # 조기 종료 조건 확인
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        counter = 0  # 카운터 초기화
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break


# In[91]:


# holdout 데이터로 MAE 측정
model.eval()
test_mae = 0
with torch.no_grad():
    test_predictions_list = []
    test_target_list = []
    for batch_data_cont, batch_data_cat, batch_target in tqdm(holdout_loader):
        test_predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
        test_predictions_list.append(test_predictions)
        test_target_list.append(batch_target)
    
    # holdout 데이터에서 MAE 측정
    test_predictions_all = torch.cat(test_predictions_list).cpu().numpy()
    test_target_all = torch.cat(test_target_list).cpu().numpy()
    test_mae = mean_absolute_error(np.expm1(test_target_all), np.expm1(test_predictions_all))

print(f'Holdout MAE: {test_mae:.4f}')


# # 하이퍼 파라미터 튜닝

#  (학습속도가 너무 느려서 엄두도 안남)

# In[ ]:


model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    n_blocks=3,
    d_block=192,
    attention_n_heads=8,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
)


# In[ ]:


# Optuna 시각화
optuna.visualization.plot_optimization_history(study)
plt.show()
optuna.visualization.plot_param_importances(study)
plt.show()


# # holdout 검증

# In[ ]:


# holdout 데이터로 MAE 측정
model.eval()
test_mae = 0
with torch.no_grad():
    test_predictions_list = []
    test_target_list = []
    for batch_data_cont, batch_data_cat, batch_target in tqdm(holdout_loader):
        test_predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
        test_predictions_list.append(test_predictions)
        test_target_list.append(batch_target)
    
    # holdout 데이터에서 MAE 측정
    test_predictions_all = torch.cat(test_predictions_list).cpu().numpy()
    test_target_all = torch.cat(test_target_list).cpu().numpy()
    test_mae = mean_absolute_error(torch.expm1(test_target_all), torch.expm1(test_predictions_all))

print(f'Test MAE: {test_mae:.4f}')


# # 재학습 후 output 생성 (제출용)

# In[ ]:


# train, test split
train_data = df[df["_type"] == "train"]
test_data = df[df["_type"] == "test"]

X_train_full = train_data.drop('deposit', axis=1)
y_train_full = train_data['deposit']
X_holdout = holdout_data.drop('deposit', axis=1)
y_holdout = holdout_data['deposit']
X_test = test.copy()

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED
)


# In[ ]:


from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
pt = PowerTransformer(method='yeo-johnson')
pt = RobustScaler()

X_train_cont = torch.Tensor(pt.fit_transform(X_train[continuous_columns])).to(device)
X_val_cont = torch.Tensor(pt.transform(X_val[continuous_columns])).to(device)
X_test_cont = torch.Tensor(pt.transform(X_test[continuous_columns])).to(device)

X_train_cat = torch.Tensor(X_train[categorical_columns].values).long().to(device)
X_val_cat = torch.Tensor(X_val[categorical_columns].values).long().to(device)
X_test_cat = torch.Tensor(X_test[categorical_columns].values).long().to(device)

y_train_log = torch.Tensor(np.log1p(y_train).values).to(device)
y_val_log = torch.Tensor(np.log1p(y_val).values).to(device)


# In[ ]:


train_dataset = TensorDataset(X_train_cont, X_train_cat, y_train_log)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_cont, X_val_cat, y_val_log)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_cont, X_test_cat)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


# # 최적의 모델로
# best_params = trial.params
# learning_rate = best_params["learning_rate"]
# batch_size = best_params["batch_size"]
# num_heads = best_params["num_heads"]
# num_attn_blocks = best_params["num_attn_blocks"]
# dropout = best_params["dropout"]


# In[ ]:


d_out = 1 # 회귀니까 1

default_kwargs = FTTransformer.get_default_kwargs() # 기본 파라미터
model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    **default_kwargs,
).to(device)
criterion = nn.L1Loss()
optimizer = model.make_default_optimizer()

# 조기 종료 설정
best_val_loss = float('inf')
patience = 5  # 조기 종료를 위한 허용 에포크 수
counter = 0


# In[ ]:


# 학습
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss_epoch = 0  # 에포크별 손실을 누적할 변수
    
    # 배치 단위로 학습
    for batch_data_cont, batch_data_cat, batch_target in tqdm(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
        train_loss = criterion(predictions, batch_target)

        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        train_loss = criterion(torch.expm1(predictions), torch.expm1(batch_target))
        train_loss_epoch += train_loss.item()
    
    train_loss_epoch /= len(train_loader)  # 배치 평균 손실 계산
    
    # 검증
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for batch_data_cont, batch_data_cat, batch_target in tqdm(val_loader):
            val_predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
            val_loss = criterion(torch.expm1(val_predictions), torch.expm1(batch_target))
            val_loss_epoch += val_loss.item()
    
    val_loss_epoch /= len(val_loader)  # 배치 평균 손실 계산

    print(f'Epoch {epoch+1}/{num_epochs}, Train MAE: {train_loss_epoch:.4f}, Val MAE: {val_loss_epoch:.4f}')

    # 조기 종료 조건 확인
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        counter = 0  # 카운터 초기화
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break


# In[ ]:


# test 데이터 생성
model.eval()
test_mae = 0
with torch.no_grad():
    test_predictions_list = []
    for batch_data_cont, batch_data_cat in tqdm(test_loader):
        test_predictions = model(batch_data_cont, batch_data_cat).view(-1)  # 1D로 변환
        test_predictions_list.append(test_predictions)
        test_target_list.append(batch_target)

    test_predictions_all = torch.cat(test_predictions_list).cpu().numpy()


# In[ ]:


# 제출용 csv 생성
# y_test_pred = tabular_model.predict(pd.concat([X_test, pd.DataFrame({'deposit': np.nan}, index=X_holdout.index)], axis=1))
sample_submission["deposit"] = test_predictions_all
sample_submission.to_csv("output2.csv", index= False)

