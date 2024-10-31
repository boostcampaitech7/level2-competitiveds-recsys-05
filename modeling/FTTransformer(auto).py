#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[2]:


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


# In[3]:


# train, test split
train = df[df["_type"] == "train"]
test = df[df["_type"] == "test"]


# In[4]:


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

# In[8]:


X_train.columns


# In[5]:


categorical_columns = [
    'contract_type', 'complex_id'
]
continuous_columns = [
    'area_m2', 'contract_year_month', 'floor', 'latitude', 'longitude', 'age',
    'max_deposit', 'pred_deposit_per_area', 'pred_deposit'
]


# In[6]:


import torch
torch.cuda.is_available()


# In[7]:


from pytorch_tabular import available_models
available_models()


# In[ ]:


# 기본 파라미터
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig, FTTransformerConfig, TabNetModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

data_config = DataConfig(
    target=[
        "deposit"
    ],  # target should always be a list.
    continuous_cols=continuous_columns,
    categorical_cols=categorical_columns,
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    early_stopping_mode="min",
    early_stopping_patience = "3",
    min_epochs = 1,

)
optimizer_config = OptimizerConfig()

model_config = FTTransformerConfig(
    task="regression",
    loss = "L1Loss",
    metrics = ["mean_absolute_error", "mean_squared_error"],
    target_range = [(int(train_data["deposit"].min() * 0.8), int(train_data["deposit"].max() * 1.2))],
    seed = 42,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True,
    
)
tabular_model.fit(train=pd.concat([X_train, y_train], axis=1), validation=pd.concat([X_val, y_val], axis=1))


# FTTransfomer   
# valid loss : 3880   

# In[ ]:


fi = tabular_model.feature_importance()


# # 하이퍼 파라미터 튜닝

# In[ ]:


import optuna
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    num_heads = trial.suggest_int("num_heads", 2, 8)
    num_attn_blocks = trial.suggest_int("num_attn_blocks", 1, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Set up data config (you can add continuous_cols and categorical_cols as needed)
    data_config = DataConfig(
        target=["deposit"],
        continuous_cols=continuous_columns,
        categorical_cols=categorical_columns,
    )
    
    # Set up trainer config
    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=100,
        early_stopping_mode="min",
        early_stopping_patience=3,
        min_epochs=1,
    )
    
    # Set up optimizer config with tuned learning rate
    optimizer_config = OptimizerConfig(lr=learning_rate)
    
    # Set up model config with tuned parameters
    model_config = TabTransformerConfig(
        task="regression",
        loss="L1Loss",
        metrics=["mean_absolute_error", "mean_squared_error"],
        target_range=[(int(train_data["deposit"].min() * 0.8), int(train_data["deposit"].max() * 1.2))],
        seed=42,
        num_heads=num_heads,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
    )
    
    # Build the model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=False,
    )
    
    # Train the model
    tabular_model.fit(train=pd.concat([X_train, y_train], axis=1), validation=pd.concat([X_val, y_val], axis=1))
    
    # Evaluate on validation set and return validation metric (e.g., MAE)
    result = tabular_model.evaluate(val=pd.concat([X_holdout, y_holdout], axis=1)) ## holdout에 대한 MSE
    
    # Optuna will minimize the objective, so return the validation MAE
    return result['valid_mean_absolute_error']

# Create a study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터 출력
print("Best trial:")
trial = study.best_trial
print(f"  MAE: {trial.value:.4f}")
print("  Best hyperparameters: ", trial.params)

# 스터디 결과 저장
joblib.dump(study, 'optuna_study_2.pkl')


# In[ ]:


# Optuna 시각화
optuna.visualization.plot_optimization_history(study)
plt.show()
optuna.visualization.plot_param_importances(study)
plt.show()


# # holdout 검증

# In[ ]:


holdout_pred = tabular_model.predict(pd.concat([X_holdout, pd.DataFrame({'deposit': np.nan}, index=X_holdout.index)], axis=1))
holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
holdout_rmse = root_mean_squared_error(y_holdout, holdout_pred)

print("Holdout 데이터셋 성능")
print(f"LightGBM MAE: {holdout_mae:.2f}")
print(f"LightGBM RMSE: {holdout_rmse:.2f}")


# ### Tab-Transformer
# Holdout 데이터셋 성능   
# LightGBM MAE: 5099.32   
# LightGBM RMSE: 8456.41   
# 
# ### FT-Transformer
# Holdout 데이터셋 성능   
# LightGBM MAE: 5099.32   
# LightGBM RMSE: 8456.41   

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


# 최적의 모델로
best_params = trial.params
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
num_heads = best_params["num_heads"]
num_attn_blocks = best_params["num_attn_blocks"]
dropout = best_params["dropout"]


# In[ ]:


# 최적의 모델로 재학습
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig, TabNetModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

data_config = DataConfig(
    target=[
        "deposit"
    ],  # target should always be a list.
    continuous_cols=continuous_columns,
    categorical_cols=categorical_columns,
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=batch_size,
    max_epochs=100,
    early_stopping_mode="min",
    early_stopping_patience = "3",
    min_epochs = 1,

)
optimizer_config = OptimizerConfig()

model_config = TabTransformerConfig(
        task="regression",
        loss="L1Loss",
        metrics=["mean_absolute_error", "mean_squared_error"],
        target_range=[(int(train_data["deposit"].min() * 0.8), int(train_data["deposit"].max() * 1.2))],
        seed=42,
        num_heads=num_heads,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
    )

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True,
    
)
tabular_model.fit(train=pd.concat([X_train, y_train], axis=1), validation=pd.concat([X_val, y_val], axis=1))
# result = tabular_model.evaluate(test)
# pred_df = tabular_model.predict(test)


# In[ ]:


# 제출용 csv 생성
y_test_pred = tabular_model.predict(pd.concat([X_test, pd.DataFrame({'deposit': np.nan}, index=X_holdout.index)], axis=1))
sample_submission["deposit"] = y_test_pred
sample_submission.to_csv("output2.csv", index= False)

