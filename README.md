![header](https://capsule-render.vercel.app/api?type=waving&color=0:EDDFE0,100:B7B7B7&width=max&height=250&section=header&text=수도권아파트-전세가예측모델&desc=RecSys05-오곡밥&fontSize=40&fontColor=4A4947&&fontAlignY=40)

## 🍚 팀원 소개

|문원찬|안규리|오소영|오준혁|윤건욱|황진욱|
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a29cbbd9-0cde-495a-bd7e-90f20759f3d1" width="100"/> | <img src="https://github.com/user-attachments/assets/c619ed82-03f3-4d48-9bba-dd60408879f9" width="100"/> | <img src="https://github.com/user-attachments/assets/1b0e54e6-57dc-4c19-97f5-69b7e6f3a9b4" width="100"/> | <img src="https://github.com/user-attachments/assets/67d19373-8cac-4676-bde1-b0637921cf7f" width="100"/> | <img src="https://github.com/user-attachments/assets/f91dd46e-9f1a-42e7-a939-db13692f4098" width="100"/> | <img src="https://github.com/user-attachments/assets/69bbb039-752e-4448-bcaa-b8a65015b778" width="100"/> |
| [![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/WonchanMoon)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/notmandarin)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/irrso)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/ojunhyuk99)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/YoonGeonWook)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/hw01931)|

</br>

## 💡 프로젝트 개요

### 프로젝트 소개
수도권 아파트 전세 실거래가를 예측하는 AI 모델을 개발하는 프로젝트입니다. 한국의 아파트 전세 시장은 부동산 정책과 밀접하게 연관되어 있어, 이를 예측하는 모델은 중요한 의미를 가집니다.

### 데이터 소개
데이터셋은 아파트의 면적, 계약 날짜, 층수, 건축 연도 등과 함께 지하철, 학교, 공원 위치 정보, 금리 정보 등 다양한 특성으로 구성되어 있습니다.

### 데이터셋 구성
- **train.csv**: 학습용 데이터 (약 180만 건)
- **test.csv**: 예측용 테스트 데이터 (약 15만 건)
- **subwayInfo.csv**: 지하철 위치 정보
- **interestRate.csv**: 금리 정보
- **schoolInfo.csv**: 학교 위치 정보
- **parkInfo.csv**: 공원 위치 정보

### 평가 방식
- **평가 지표**: Mean Absolute Error (MAE)를 사용하여 예측 성능을 평가합니다.

### 프로젝트 목표
데이터를 기반으로 아파트 전세 실거래가를 정확히 예측해, 부동산 시장의 정보 비대칭성을 해소하고자 합니다.

</br>

## 📑랩업 리포트
```
정리전
```

</br>

## 📂폴더구조
```
# level2-competitiveds-recsys-05/
│
├── .github/
│   └── .keep
│
├── code/
│   └── requirements.txt
│
├── feat_eng/
│   ├── __pycache__/
│   │   ├── feat_eng.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   ├── FE_gw.ipynb                     # 기본 데이터 전처리 및 POI 데이터를 이용한 피처 생성
│   ├── feat_eng.py                     # 주요 피처 엔지니어링 로직 (단지 ID 생성, POI 거리 등)
│   ├── feat_eng2.ipynb                 # 클러스터 기반 Prophet 모델을 이용한 피처 생성
│   └── utils.py                        # 메모리 사용량 최적화 유틸리티 함수
│
├── modeling/
│   ├── CatBoost.ipynb                  # CatBoost 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
│   ├── Ensemble.ipynb                  # LGBM, XGBoost, CatBoost 모델을 앙상블한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화, 최종 제출 코드
│   ├── FTTransformer(auto).ipynb       # pytorch_tabular를 통해 TF-Transformer 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
│   ├── FTTransformer(torch).ipynb      # rtdl_revisiting_models를 통해 TF-Transformer 모델을 사용한 전세가 예측
│   ├── LGBM.ipynb                      # LGBM 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
│   ├── LGMN_deposit_piecewise.ipynb    # LGBM 모델을 사용한 전세가 구간별 예측
│   ├── Stacking_Ensemble.ipynb         # LGBM, XGBoost, CatBoost 모델을 OOF 스태킹한 전세가 예측
│   ├── TabNet.ipynb                    # TabNet 모델을 사용한 전세가 예측
│   ├── XGBoost.ipynb                   # XGBoost 모델을 사용한 전세가 예측 및 Optuna를 통한 하이퍼파라미터 최적화
│   └── cluster_piecewise.ipynb         # LGBM, ElasticNet 모델을 클러스터별 piecewise 예측
│
├── .gitignore
└── README.md
```
</br>

## ⚙️ 개발 환경
#### OS: Linux (5.4.0-99-generic, x86_64)
#### GPU: Tesla V100-SXM2-32GB (CUDA Version: 12.2)
#### CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 8 Cores
</br>

## 🔧 기술 스택

#### 프로그래밍 언어 <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=python&logoColor=white"/>

#### 데이터 분석 및 전처리 <img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=scipy&logoColor=white"/>


#### 모델 학습 및 평가 <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat-square&logo=keras&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/LightGBM-41454A.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/XGBoost-1578D3.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/CatBoost-FECC00.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch--tabular-EE4C2C.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/tab--transformer--pytorch-EE4C2C.svg?style=flat-square&logoColor=white"/>

  
#### 시각화 도구 <img src="https://img.shields.io/badge/Matplotlib-3F4F75.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/seaborn-221E68.svg?style=flat-square&logoColor=white"/>

#### 모델 최적화 및 클러스터링 <img src="https://img.shields.io/badge/Optuna-13448F.svg?style=flat-square&logo=optuna&logoColor=white"/> <img src="https://img.shields.io/badge/HDBSCAN-F7931E.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/Pmdarima-3775A9.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/umap--learn-34567C.svg?style=flat-square&logoColor=white"/>

#### 개발 환경 <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=jupyter&logoColor=white"/>

#### 실험 관리 및 추적 <img src="https://img.shields.io/badge/Weights&Biases-FFBE00.svg?style=flat-square&logo=weightsandbiases&logoColor=black"/>

#### 기타 유틸리티 <img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black"/> <img src="https://img.shields.io/badge/OpenPyXL-013243.svg?style=flat-square&logoColor=white"/>
