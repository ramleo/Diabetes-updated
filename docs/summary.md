# Diabetes-updated — ML Pipeline Summary

**Project:** Diabetes prediction (binary classification)
**Generated:** 2026-05-26 | **Python:** 3.14.2 | **random_state:** 42

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| File | `data/diabetes.csv` |
| Shape | 768 rows × 9 columns |
| Features | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age |
| Target | `Outcome` (0 = No Diabetes, 1 = Diabetes) |
| Dtypes | All numeric (int64 / float64) |
| Missing values | 0 explicit NaN; zero-proxy missingness in 5 columns |
| Duplicate rows | 0 |

**Class balance:**
| Class | Count | % |
|---|---|---|
| 0 — No Diabetes | 500 | 65.1% |
| 1 — Diabetes | 268 | 34.9% |

---

## 2. Exploratory Data Analysis

### Zero-Proxy Missing Values
| Column | Zero Count | % |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

### Top Correlations with Outcome
| Feature | Pearson r |
|---|---|
| Glucose | 0.467 |
| BMI | 0.293 |
| Age | 0.238 |

### Outlier Summary (IQR method)
| Column | Outlier Count |
|---|---|
| BloodPressure | 45 |
| Insulin | 34 |
| DiabetesPedigreeFunction | 29 |
| BMI | 19 |
| Pregnancies | 4 |

### Notable Distribution Insights
- **Insulin** is the most problematic column: 48.7% zeros + extreme right skew (max=846, std=115)
- **SkinThickness** is heavily zero-inflated (29.6%)
- **DiabetesPedigreeFunction** is right-skewed (range 0.078–2.42)
- **Age** is right-skewed; most patients 21–40, outliers up to 81

### Saved Plots
| File | Description |
|---|---|
| `plots/class_distribution.png` | Bar chart of Outcome counts |
| `plots/feature_distributions.png` | Histograms for all 8 features |
| `plots/correlation_heatmap.png` | Heatmap of correlation matrix |
| `plots/boxplots.png` | Feature boxplots grouped by Outcome |

---

## 3. Preprocessing Pipeline

**Steps applied:**
1. Replace zeros with NaN in: Glucose, BloodPressure, SkinThickness, Insulin, BMI
2. `SimpleImputer(strategy='median')` — fit on train only
3. `StandardScaler` — fit on train only
4. Encode target with `LabelEncoder`

**Train-Test Split:** 80/20 stratified on `Outcome`, `random_state=42`
| Split | Rows | Class 0 | Class 1 |
|---|---|---|---|
| Train | 614 | 400 (65.1%) | 214 (34.9%) |
| Test | 154 | 100 (64.9%) | 54 (35.1%) |

---

## 4. Model Selection & Hyperparameter Tuning

GridSearchCV | `cv=5` | `scoring='accuracy'` | `n_jobs=-1`

| Model | Best Params | CV Accuracy |
|---|---|---|
| **LogisticRegression** ⭐ | C=1 | **0.7818** |
| RandomForestClassifier | max_depth=5, n_estimators=100 | 0.7721 |
| SVC | C=1, kernel=linear | 0.7704 |
| GradientBoostingClassifier | lr=0.05, max_depth=3, n_estimators=100 | 0.7671 |

---

## 5. Model Evaluation

**Best model:** LogisticRegression (C=1, solver=saga)
**Test accuracy:** 70.78%

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Diabetes (0) | 0.75 | 0.82 | 0.78 | 100 |
| Diabetes (1) | 0.60 | 0.50 | 0.55 | 54 |
| **Weighted avg** | **0.70** | **0.71** | **0.70** | **154** |

---

## 6. Final Pipeline Architecture

```
Input (raw DataFrame, 8 features)
    │
    ▼
ColumnTransformer
  └─ numeric: SimpleImputer(strategy='median') → StandardScaler
    │
    ▼
LogisticRegression(C=1, solver='saga', random_state=42)
    │
    ▼
Prediction: 0 (No Diabetes) | 1 (Diabetes) + predict_proba
```

---

## 7. Artifacts

| File | Description |
|---|---|
| `data/diabetes.csv` | Raw dataset (768 × 9) |
| `src/preprocess.py` | Preprocessing script |
| `models/preprocessor.pkl` | Fitted ColumnTransformer |
| `models/X_train_raw.pkl` | Raw train features (614 × 8) |
| `models/X_test_raw.pkl` | Raw test features (154 × 8) |
| `models/y_train.npy` | Encoded train labels (614,) |
| `models/y_test.npy` | Encoded test labels (154,) |
| `models/label_encoder.pkl` | Fitted LabelEncoder (classes: [0, 1]) |
| `models/X_train.npy` | Preprocessed train array |
| `models/X_test.npy` | Preprocessed test array |
| `models/final_pipeline.pkl` | Full production pipeline |
| `app.py` | FastAPI prediction API |
| `plots/*.png` | EDA visualisations (4 files) |
| `docs/summary.md` | This document |

---

## 8. Reproducibility

```python
import joblib
import pandas as pd

# Load the final pipeline
pipeline = joblib.load('models/final_pipeline.pkl')

# Single prediction
sample = pd.DataFrame([{
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,       # 0 → treated as NaN → imputed
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}])

pred = pipeline.predict(sample)
prob = pipeline.predict_proba(sample)
label = 'Diabetes' if pred[0] == 1 else 'No Diabetes'
print(f"Prediction: {label}  |  P(diabetes) = {prob[0][1]:.4f}")
# → Prediction: Diabetes  |  P(diabetes) = 0.7059
```
