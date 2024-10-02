import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load your models (assumes models are trained and saved)
lgb_model = lgb.Booster(model_file='/data/ephemeral/home/level2-competitiveds-recsys-05/code/lightgbm_model.txt')
lasso_model = Lasso()  # Load your trained Lasso model
lasso_model = joblib.load('/data/ephemeral/home/level2-competitiveds-recsys-05/code/lasso_model.pkl')  # Assuming saved as a pickle
ridge_model = Ridge()  # Load your trained Ridge model
ridge_model = joblib.load('/data/ephemeral/home/level2-competitiveds-recsys-05/code/ridge_model.pkl')  # Assuming saved as a pickle
# rf_model = RandomForestRegressor()  # Load your trained Random Forest model
# rf_model.load('rf_model.pkl')  # Assuming saved as a pickle
xgb_model = xgb.XGBRegressor()  # Load your trained XGBoost model
xgb_model = joblib.load('/data/ephemeral/home/level2-competitiveds-recsys-05/code/xgboost_model.pkl')  # Assuming saved as a pickle
lr_model = LinearRegression()  # Load your trained Linear Regression model
lr_model = joblib.load('/data/ephemeral/home/level2-competitiveds-recsys-05/code/linear_regression_model.pkl')  # Assuming saved as a pickle

# Streamlit UI setup
st.title("부동산 전세가 예측")

# User input fields
area = st.number_input("면적 (m²)", min_value=1.0, max_value=10000.0, value=50.0)
contract_year_month = st.number_input("계약 연도 및 월 (예: 202307)", min_value=202001, max_value=202312, value=202307)
contract_day = st.number_input("계약 일", min_value=1, max_value=31, value=1)
contract_type = st.selectbox("계약 유형", options=["신규", "갱신", "모름"])
floor = st.number_input("층", min_value=0, max_value=50, value=1)
latitude = st.number_input("위도", min_value=36.0, max_value=38.5, value=37.5)
longitude = st.number_input("경도", min_value=126.0, max_value=128.0, value=127.0)

# Predict button
if st.button("예측하기"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'area_m2': [area],
        'contract_year_month': [contract_year_month],
        'contract_day': [contract_day],
        'contract_type': [contract_type],
        'floor': [floor],
        'latitude': [latitude],
        'longitude': [longitude]
    })
    
    # Process the input data as necessary
    # For example, convert categorical variables to numerical
    input_data['contract_type'] = input_data['contract_type'].map({'신규': 0, '갱신': 1, '모름': 2})

    # Make predictions
    lgb_pred = lgb_model.predict(input_data)
    lasso_pred = lasso_model.predict(input_data)
    ridge_pred = ridge_model.predict(input_data)
    # rf_pred = rf_model.predict(input_data)
    xgb_pred = xgb_model.predict(input_data)
    lr_pred = lr_model.predict(input_data)

    # Display the results
    st.subheader("예측 결과")
    st.write(f"LightGBM 예측: {lgb_pred[0]:.2f} 원")
    st.write(f"Lasso 예측: {lasso_pred[0]:.2f} 원")
    st.write(f"Ridge 예측: {ridge_pred[0]:.2f} 원")
    # st.write(f"Random Forest 예측: {rf_pred[0]:.2f} 원")
    st.write(f"XGBoost 예측: {xgb_pred[0]:.2f} 원")
    st.write(f"Linear Regression 예측: {lr_pred[0]:.2f} 원")