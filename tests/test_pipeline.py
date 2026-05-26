"""
Diabetes ML Pipeline - Test Suite
16 automated checks covering artifact integrity, predictions, accuracy, and robustness.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PIPELINE_PATH    = os.path.join(ROOT, "models", "final_pipeline.pkl")
LABEL_ENC_PATH   = os.path.join(ROOT, "models", "label_encoder.pkl")
PREPROCESSOR_PATH= os.path.join(ROOT, "models", "preprocessor.pkl")
X_TEST_RAW_PATH  = os.path.join(ROOT, "models", "X_test_raw.pkl")
Y_TEST_PATH      = os.path.join(ROOT, "models", "y_test.npy")
ACCURACY_THRESHOLD = 0.68

passed = 0
failed = 0
results = []

def record(test_num, desc, ok, detail=""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] Test {test_num:02d}: {desc}"
    if detail:
        msg += f" — {detail}"
    results.append(msg)
    if ok:
        passed += 1
    else:
        failed += 1

# ── Test 1: final_pipeline.pkl exists and loads ───────────────────────────────
try:
    assert os.path.exists(PIPELINE_PATH), "File not found"
    import joblib
    pipeline = joblib.load(PIPELINE_PATH)
    record(1, "final_pipeline.pkl exists and loads without error", True)
except Exception as e:
    record(1, "final_pipeline.pkl exists and loads without error", False, str(e))
    pipeline = None

# ── Test 2: label_encoder.pkl exists and loads ────────────────────────────────
try:
    assert os.path.exists(LABEL_ENC_PATH), "File not found"
    le = joblib.load(LABEL_ENC_PATH)
    record(2, "label_encoder.pkl exists and loads", True)
except Exception as e:
    record(2, "label_encoder.pkl exists and loads", False, str(e))
    le = None

# ── Test 3: preprocessor.pkl exists and loads ────────────────────────────────
try:
    assert os.path.exists(PREPROCESSOR_PATH), "File not found"
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    record(3, "preprocessor.pkl exists and loads", True)
except Exception as e:
    record(3, "preprocessor.pkl exists and loads", False, str(e))
    preprocessor = None

# ── Test 4: Pipeline has exactly 2 steps: 'preprocessor' and 'model' ─────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    steps = list(pipeline.named_steps.keys())
    assert steps == ['preprocessor', 'model'], f"Got steps: {steps}"
    record(4, "Pipeline has exactly 2 steps: 'preprocessor' and 'model'", True)
except Exception as e:
    record(4, "Pipeline has exactly 2 steps: 'preprocessor' and 'model'", False, str(e))

# ── Test 5: Pipeline model is LogisticRegression ──────────────────────────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    from sklearn.linear_model import LogisticRegression
    model = pipeline.named_steps['model']
    assert isinstance(model, LogisticRegression), f"Got: {type(model).__name__}"
    record(5, "Pipeline model is a LogisticRegression", True)
except Exception as e:
    record(5, "Pipeline model is a LogisticRegression", False, str(e))

# ── Test 6: Label encoder has exactly 2 classes: [0, 1] ──────────────────────
try:
    assert le is not None, "Label encoder not loaded"
    classes = list(le.classes_)
    assert classes == [0, 1], f"Got classes: {classes}"
    record(6, "Label encoder has exactly 2 classes: [0, 1]", True)
except Exception as e:
    record(6, "Label encoder has exactly 2 classes: [0, 1]", False, str(e))

# ── Test 7: Single high-risk sample → expected prediction 1 ──────────────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    sample_high = pd.DataFrame([{
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72,
        'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }])
    pred = pipeline.predict(sample_high)[0]
    assert pred == 1, f"Expected 1, got {pred}"
    record(7, "Single-sample high-risk prediction returns 1", True)
except Exception as e:
    record(7, "Single-sample high-risk prediction returns 1", False, str(e))

# ── Test 8: predict_proba sums to ~1.0 for single sample ─────────────────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    sample = pd.DataFrame([{
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72,
        'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }])
    proba = pipeline.predict_proba(sample)[0]
    total = sum(proba)
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"Probabilities sum to {total}"
    record(8, "predict_proba returns probabilities summing to ~1.0", True)
except Exception as e:
    record(8, "predict_proba returns probabilities summing to ~1.0", False, str(e))

# ── Test 9: Single low-risk sample → expected prediction 0 ───────────────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    sample_low = pd.DataFrame([{
        'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66,
        'SkinThickness': 29, 'Insulin': 0, 'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351, 'Age': 31
    }])
    pred = pipeline.predict(sample_low)[0]
    assert pred == 0, f"Expected 0, got {pred}"
    record(9, "Single-sample low-risk prediction returns 0", True)
except Exception as e:
    record(9, "Single-sample low-risk prediction returns 0", False, str(e))

# ── Test 10: Batch prediction on X_test_raw returns shape (154,) ─────────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    assert os.path.exists(X_TEST_RAW_PATH), "X_test_raw.pkl not found"
    X_test_raw = joblib.load(X_TEST_RAW_PATH)
    batch_preds = pipeline.predict(X_test_raw)
    assert batch_preds.shape == (154,), f"Got shape: {batch_preds.shape}"
    record(10, "Batch prediction on X_test_raw returns correct shape (154,)", True)
except Exception as e:
    record(10, "Batch prediction on X_test_raw returns correct shape (154,)", False, str(e))
    batch_preds = None
    X_test_raw = None

# ── Test 11: All batch predictions are in {0, 1} ─────────────────────────────
try:
    assert batch_preds is not None, "Batch predictions not available"
    unique_vals = set(batch_preds.tolist())
    assert unique_vals.issubset({0, 1}), f"Unexpected values: {unique_vals - {0, 1}}"
    record(11, "All batch predictions are in {0, 1}", True)
except Exception as e:
    record(11, "All batch predictions are in {0, 1}", False, str(e))

# ── Test 12: Test set accuracy meets threshold (>= 0.68) ─────────────────────
accuracy = None
try:
    assert batch_preds is not None, "Batch predictions not available"
    assert os.path.exists(Y_TEST_PATH), "y_test.npy not found"
    y_test = np.load(Y_TEST_PATH)
    accuracy = float(np.mean(batch_preds == y_test))
    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}"
    record(12, f"Test set accuracy meets threshold >= {ACCURACY_THRESHOLD}", True,
           f"accuracy={accuracy:.4f}")
except Exception as e:
    record(12, f"Test set accuracy meets threshold >= {ACCURACY_THRESHOLD}", False, str(e))

# ── Test 13: Per-class recall: class 0 >= 0.70, class 1 >= 0.40 ──────────────
try:
    assert batch_preds is not None, "Batch predictions not available"
    y_test = np.load(Y_TEST_PATH)
    mask0 = (y_test == 0)
    mask1 = (y_test == 1)
    recall0 = float(np.mean(batch_preds[mask0] == 0))
    recall1 = float(np.mean(batch_preds[mask1] == 1))
    assert recall0 >= 0.70, f"Class 0 recall {recall0:.4f} < 0.70"
    assert recall1 >= 0.40, f"Class 1 recall {recall1:.4f} < 0.40"
    record(13, "Per-class recall: class 0 >= 0.70, class 1 >= 0.40", True,
           f"recall_0={recall0:.4f}, recall_1={recall1:.4f}")
except Exception as e:
    record(13, "Per-class recall: class 0 >= 0.70, class 1 >= 0.40", False, str(e))

# ── Test 14: Probability outputs are in [0, 1] for all test samples ───────────
try:
    assert pipeline is not None, "Pipeline not loaded"
    assert X_test_raw is not None, "X_test_raw not loaded"
    probas = pipeline.predict_proba(X_test_raw)
    assert probas.min() >= 0.0, f"Min probability {probas.min()} < 0"
    assert probas.max() <= 1.0, f"Max probability {probas.max()} > 1"
    record(14, "Probability outputs are in [0, 1] range for all test samples", True)
except Exception as e:
    record(14, "Probability outputs are in [0, 1] range for all test samples", False, str(e))

# ── Test 15: Consistency check — same input always yields same prediction ─────
try:
    assert pipeline is not None, "Pipeline not loaded"
    sample = pd.DataFrame([{
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72,
        'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }])
    preds = [pipeline.predict(sample)[0] for _ in range(5)]
    assert len(set(preds)) == 1, f"Got inconsistent predictions: {preds}"
    record(15, "Consistency check: same input always yields same prediction", True)
except Exception as e:
    record(15, "Consistency check: same input always yields same prediction", False, str(e))

# ── Test 16: Robust to None/NaN for nullable columns (Insulin, SkinThickness) ─
try:
    assert pipeline is not None, "Pipeline not loaded"
    sample_nan = pd.DataFrame([{
        'Pregnancies': 3, 'Glucose': 120, 'BloodPressure': 70,
        'SkinThickness': None, 'Insulin': None, 'BMI': 28.0,
        'DiabetesPedigreeFunction': 0.5, 'Age': 40
    }])
    pred = pipeline.predict(sample_nan)[0]
    assert pred in {0, 1}, f"Got unexpected value: {pred}"
    record(16, "Pipeline is robust to None/NaN inputs for nullable columns", True)
except Exception as e:
    record(16, "Pipeline is robust to None/NaN inputs for nullable columns", False, str(e))

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  DIABETES ML PIPELINE — TEST RESULTS")
print("=" * 65)
for line in results:
    print(line)
print("-" * 65)
if accuracy is not None:
    print(f"  Overall test-set accuracy : {accuracy:.4f}")
print(f"  Tests passed              : {passed}/16")
print(f"  Tests failed              : {failed}/16")
print("=" * 65)

if failed > 0:
    sys.exit(1)
