import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np


# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# MAE 저장 리스트
mae_results = []

pred_results = []

test_data['predict_deposit'] = np.nan

# 클러스터별 모델 학습 및 검증
for cluster in train_data['cluster_labels2'].unique():
    # 클러스터별 train 데이터 분리
    cluster_train = train_data[train_data['cluster_labels2'] == cluster]
    
    # 'deposit'을 종속 변수로 설정하고 나머지는 독립 변수로 사용
    X_cluster = cluster_train.drop(columns=['deposit'])
    y_cluster = cluster_train['deposit']

    # 교차검증을 위한 fold별 학습 및 검증
    fold_mae = []
    for train_index, val_index in kf.split(X_cluster):
        X_train_fold, X_val_fold = X_cluster.iloc[train_index], X_cluster.iloc[val_index]
        y_train_fold, y_val_fold = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
        
        # LGBM 모델 학습
        model = lgb.LGBMRegressor()
        model.fit(X_train_fold, y_train_fold)
        
        # 검증 데이터에 대한 예측
        y_val_pred = model.predict(X_val_fold)
        
        # MAE 계산
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        fold_mae.append(mae)
    
    # 클러스터별 평균 MAE 저장
    cluster_mae = np.mean(fold_mae)
    cluster_size = len(cluster_train)
    mae_results.append({'cluster_label': cluster, 'mae': cluster_mae, 'size': cluster_size})
    print(f"Cluster {cluster} - Cross-Validation MAE: {cluster_mae}")

    # 학습 완료된 모델로 test 데이터에 대한 예측 수행
    
    cluster_test = test_data[test_data['cluster_labels2'] == cluster]

    if not cluster_test.empty:
        X_test_cluster = cluster_test[columns_needed_test]
        # 예측 수행
        test_preds = model.predict(X_test_cluster)
        # 예측값 저장
        test_data.loc[cluster_test.index, 'predict_deposit'] = test_preds    

mae_df = pd.DataFrame(mae_results)
pred_df = pd.DataFrame(pred_results)

# 전체 MAE 평균 출력 (train 데이터에서의 교차 검증 결과)
overall_mae = mae_df['mae'].mean()
print(f"Overall Cross-Validation MAE: {overall_mae}")

# test_data에 예측값 할당 완료 후 결과 출력
# print(test_data[['contract_year_month', 'predicted_deposit']])

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet


feature_columns = [col for col in train_data.columns if col not in ['deposit', 'cluster_labels2']]

# MAE 저장 리스트
mae_scores = []
cluster_list = sorted(train_data['cluster_labels2'].unique())

# 테스트 데이터에 예측값을 저장할 컬럼 생성
test_data['predict_deposit'] = np.nan

# 클러스터별 모델 학습 및 검증
for cluster in cluster_list:
    print(f"\nProcessing Cluster {cluster}")
    
    # 클러스터별 train 데이터 분리
    cluster_train = train_data[train_data['cluster_labels2'] == cluster]
    
    # 데이터 크기 확인 (데이터가 너무 적은 경우 제외할 수 있음)
    if len(cluster_train) < 10:
        print(f"Cluster {cluster} has insufficient data for training.")
        continue
    
    # 독립 변수(X)와 종속 변수(y) 분리
    X_cluster = cluster_train[feature_columns]
    y_cluster = cluster_train['deposit']
    
    # 교차검증을 위한 fold별 학습 및 검증
    fold_mae = []
    # 하이퍼파라미터 그리드 설정
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    
    for train_index, val_index in kf.split(X_cluster):
        X_train_fold, X_val_fold = X_cluster.iloc[train_index], X_cluster.iloc[val_index]
        y_train_fold, y_val_fold = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
        
        # GridSearchCV를 통한 하이퍼파라미터 튜닝
        model = ElasticNet(max_iter=10000)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=3,  # 내부 교차 검증 설정
            n_jobs=-1
        )
        grid_search.fit(X_train_fold, y_train_fold)
        
        # 최적의 모델로 예측
        best_model = grid_search.best_estimator_
        y_val_pred = best_model.predict(X_val_fold)
        
        # MAE 계산
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        fold_mae.append(mae)
    
    # 클러스터별 평균 MAE 저장
    cluster_mae = np.mean(fold_mae)
    mae_scores.append(cluster_mae)
    print(f"Cluster {cluster} - Cross-Validation MAE: {cluster_mae:.4f}")
    print(f"Best params for Cluster {cluster}: {grid_search.best_params_}")
    
    # 전체 클러스터 데이터를 사용하여 최종 모델 재학습
    best_model.fit(X_cluster, y_cluster)
    
    # 해당 클러스터의 테스트 데이터에 대한 예측
    cluster_test = test_data[test_data['cluster_labels2'] == cluster]
    if not cluster_test.empty:
        X_test_cluster = cluster_test[feature_columns]
        # 예측 수행
        test_preds = best_model.predict(X_test_cluster)
        # 예측값 저장
        test_data.loc[cluster_test.index, 'predict_deposit'] = test_preds

