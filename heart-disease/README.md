# Heart Disease Prediction

Kaggle 대회 참가 프로젝트. 환자 임상 데이터를 기반으로 심장병 여부(Presence / Absence)를 예측하는 이진 분류 문제.

---

## 데이터셋

- **출처**: Kaggle (Heart Disease Dataset)
- **규모**: Train 630,000건 / Test 270,000건
- **피처**: Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels fluro, Thallium (13개)
- **타겟**: Heart Disease (Presence=1 / Absence=0)

---

## 프로젝트 구조

```
heart-disease/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── notebooks/
    ├── 01_eda.ipynb          # 탐색적 데이터 분석
    ├── 02_fe_modeling.ipynb  # 피처 엔지니어링 + 단일 모델 + SHAP 분석
    └── 03_final.ipynb        # 하이퍼파라미터 튜닝 + 스태킹 앙상블
```

---

## 분석 과정

### 1. EDA (`01_eda.ipynb`)
- 클래스 분포 확인 (Presence / Absence 비율)
- 범주형 피처별 심장병 비율 시각화 (이중 축 차트)
- 연속형 피처 분포 및 이상치 탐색

### 2. 피처 엔지니어링 (`02_fe_modeling.ipynb`)
EDA 인사이트를 바탕으로 도메인 기반 파생 피처 생성:

| 피처 | 설명 |
|------|------|
| `age_squared` | 나이의 제곱 (비선형 효과 반영) |
| `age_40_60` | 40~60세 구간 플래그 (고위험 연령대) |
| `chol_capped` | 콜레스테롤 상한 320 캡핑 (이상치 처리) |
| `chol_peak` | 콜레스테롤 280~320 구간 플래그 |
| `maxhr_inv` | Max HR의 역수 (비선형 관계 반영) |
| `maxhr_high` | Max HR ≥ 150 플래그 |
| `st_log` | ST depression 로그 변환 |
| `st_abnormal` | ST depression > 0 플래그 (비정상 여부) |

### 3. 모델링 및 앙상블 (`03_final.ipynb`)
- **하이퍼파라미터 튜닝**: Optuna 기반 베이지안 최적화
  - XGBoost: 100 trials → CV AUC **0.9553**
  - LightGBM: 100 trials
  - RandomForest: 50 trials
  - Logistic Regression: GridSearch
- **앙상블**: CV 기반 스태킹 (Base: XGB + LGBM + RF, Meta: Logistic)
- **최종 제출**: 단일 모델 4개 + Stacking 제출 파일 생성

---

## 주요 결과

| 모델 | ROC-AUC |
|------|--------|
| XGBoost (Optuna) | **0.955693** |
| LightGBM (Optuna) | **0.957079** |
| RandomForest (Optuna) | **0.950321** |
| Stacking Ensemble | **0.955755** |

- test 성능 : XGBoost 단일 > Stacking > LightGBM

---

## 사용 기술

- **언어**: Python 3.13
- **주요 라이브러리**: pandas, numpy, scikit-learn, XGBoost, LightGBM, Optuna, matplotlib, SHAP
- **분석 기법**: EDA, Feature Engineering, Hyperparameter Tuning (Bayesian Optimization), Stacking Ensemble, SHAP Analysis
