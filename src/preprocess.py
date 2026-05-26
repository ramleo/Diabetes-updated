"""
Preprocessing pipeline for the Pima Indians Diabetes dataset.
Task: Binary Classification (Outcome: 0/1)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ── 2. Replace zero-proxy missingness with NaN (CoW-safe) ─────────────────────
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in ZERO_AS_NAN_COLS:
    df = df.assign(**{col: df[col].astype(float).replace(0, np.nan)})

nan_counts = df[ZERO_AS_NAN_COLS].isna().sum()
print("\nNaN counts after zero-replacement:")
for col, cnt in nan_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {col}: {cnt} ({pct:.1f}%)")

# ── 3. Separate features and target ───────────────────────────────────────────
TARGET_COL   = "Outcome"
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].copy()
y_raw = df[TARGET_COL].values

# ── 4. Encode target ───────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nTarget classes: {le.classes_} → encoded as {np.unique(y)}")

# ── 5. Stratified 80/20 split ─────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nSplit shapes:")
print(f"  X_train_raw : {X_train_raw.shape}")
print(f"  X_test_raw  : {X_test_raw.shape}")
print(f"  y_train     : {y_train.shape}")
print(f"  y_test      : {y_test.shape}")

# ── 6. Class distribution ──────────────────────────────────────────────────────
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique,  test_counts  = np.unique(y_test,  return_counts=True)

print("\nClass distribution — train set:")
for cls, cnt in zip(train_unique, train_counts):
    label = le.inverse_transform([cls])[0]
    print(f"  Class {cls} ({label}): {cnt} ({cnt/len(y_train)*100:.1f}%)")

print("Class distribution — test set:")
for cls, cnt in zip(test_unique, test_counts):
    label = le.inverse_transform([cls])[0]
    print(f"  Class {cls} ({label}): {cnt} ({cnt/len(y_test)*100:.1f}%)")

# ── 7. Build ColumnTransformer (all 8 features are numeric) ───────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, FEATURE_COLS),
    ],
    remainder="drop",
)

# Fit on training data only
preprocessor.fit(X_train_raw)
print(f"\nPreprocessor fitted on X_train_raw ({X_train_raw.shape[0]} samples).")

# Transform both splits
X_train_arr = preprocessor.transform(X_train_raw)
X_test_arr  = preprocessor.transform(X_test_raw)
print(f"Preprocessed arrays — X_train: {X_train_arr.shape}, X_test: {X_test_arr.shape}")

# ── 8. Save all artifacts ──────────────────────────────────────────────────────
artifacts = {
    os.path.join(MODELS_DIR, "X_train_raw.pkl"):    X_train_raw,
    os.path.join(MODELS_DIR, "X_test_raw.pkl"):     X_test_raw,
    os.path.join(MODELS_DIR, "label_encoder.pkl"):  le,
    os.path.join(MODELS_DIR, "preprocessor.pkl"):   preprocessor,
}
for path, obj in artifacts.items():
    joblib.dump(obj, path)

np.save(os.path.join(MODELS_DIR, "y_train.npy"),  y_train)
np.save(os.path.join(MODELS_DIR, "y_test.npy"),   y_test)
np.save(os.path.join(MODELS_DIR, "X_train.npy"),  X_train_arr)
np.save(os.path.join(MODELS_DIR, "X_test.npy"),   X_test_arr)

print("\nSaved artifacts:")
all_paths = list(artifacts.keys()) + [
    os.path.join(MODELS_DIR, "y_train.npy"),
    os.path.join(MODELS_DIR, "y_test.npy"),
    os.path.join(MODELS_DIR, "X_train.npy"),
    os.path.join(MODELS_DIR, "X_test.npy"),
]
for p in all_paths:
    size_kb = os.path.getsize(p) / 1024
    print(f"  {p}  ({size_kb:.1f} KB)")

print("\nPreprocessing complete — no errors.")
