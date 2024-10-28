![header](https://capsule-render.vercel.app/api?type=waving&color=0:EDDFE0,100:B7B7B7&width=max&height=175&section=header&text=모델링&desc=수도권아파트-전세가예측모델&fontSize=30&fontColor=4A4947&&fontAlignY=40)

## 📂 모델링 개요

### modeling 디렉토리 개요

- **CatBoost.ipynb**: CatBoost 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
- **Ensemble.ipynb**: LGBM, XGBoost, CatBoost 모델을 앙상블한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화, **최종 제출 코드**
- **FTTransformer(auto).ipynb**: pytorch_tabular를 통해 TF-Transformer 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
- **FTTransformer(torch).ipynb**: rtdl_revisiting_models를 통해 TF-Transformer 모델을 사용한 전세가 예측
- **LGBM.ipynb**: LGBM 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
- **LGMN_deposit_piecewise.ipynb**: LGBM 모델을 사용한 전세가 구간별 예측
- **Stacking_Ensemble.ipynb**: LGBM, XGBoost, CatBoost 모델을 OOF 스태킹한 전세가 예측
- **TabNet.ipynb**: TabNet 모델을 사용한 전세가 예측
- **XGBoost.ipynb**: XGBoost 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
- **cluster_piecewise.ipynb**: LGBM, ElasticNet 모델을 클러스터별 piecewise 예측 
