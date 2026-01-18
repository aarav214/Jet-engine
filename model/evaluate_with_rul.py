import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==================================================
# PATH SETUP (ROBUST, NO DUPLICATION)
# ==================================================

BASE_DIR = Path(__file__).resolve().parent        # AI/model
PROJECT_ROOT = BASE_DIR.parent                   # AI/
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = BASE_DIR / "rf_model.pkl"
TEST_PATH = DATA_DIR / "test_FD001.txt"
# Try common RUL filename variants (some datasets use 'RUI' typo)
RUL_PATH = None
for name in ("RUL_FD001.txt", "RUI_FD001.txt"):
    candidate = DATA_DIR / name
    if candidate.exists():
        RUL_PATH = candidate
        break

if RUL_PATH is None:
    raise FileNotFoundError(
        f"No RUL file found in {DATA_DIR}. Tried RUL_FD001.txt and RUI_FD001.txt"
    )

# ==================================================
# LOAD MODEL
# ==================================================

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]

print("✅ Model loaded")

# ==================================================
# LOAD TEST DATA
# ==================================================

COLUMN_NAMES = ['engine_id', 'cycle',
                'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

test_df = pd.read_csv(
    TEST_PATH,
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)

print(f"✅ Test data loaded: {len(test_df)} rows")

# ==================================================
# LOAD TRUE RUL (NASA FILE)
# ==================================================

true_rul = pd.read_csv(
    RUL_PATH,
    header=None,
    names=["true_rul"]
)

print(f"✅ True RUL loaded: {len(true_rul)} engines")

# ==================================================
# GET LAST CYCLE PER ENGINE
# ==================================================

last_cycles = (
    test_df
    .sort_values("cycle")
    .groupby("engine_id")
    .tail(1)
    .reset_index(drop=True)
)

# Sanity check
assert len(last_cycles) == len(true_rul), "❌ Engine count mismatch!"

print("✅ Extracted last cycle for each engine")

# ==================================================
# PREDICT NORMALIZED RUL
# ==================================================

X_test = last_cycles[features]

pred_ratio = model.predict(X_test)
pred_ratio = np.clip(pred_ratio, 0, 1)

# ==================================================
# CONVERT RUL_RATIO → RUL (CYCLES)
# ==================================================

observed_cycles = last_cycles["cycle"].values
pred_rul = pred_ratio * observed_cycles

# ----------------------------
# Diagnostics / sanity checks
# ----------------------------
results_df = last_cycles[["engine_id", "cycle"]].copy()
results_df["pred_rul"] = pred_rul
results_df["true_rul"] = true_rul["true_rul"].values

# Basic shape checks
print(f"\nDiagnostic checks:")
print(f" - last_cycles shape: {last_cycles.shape}")
print(f" - true_rul shape: {true_rul.shape}")
print(f" - results_df shape: {results_df.shape}")

# Check engine id alignment
if len(results_df) != len(true_rul):
    raise AssertionError("Engine count mismatch between predictions and true RUL")

# Error statistics
results_df["abs_error"] = np.abs(results_df["true_rul"] - results_df["pred_rul"])
print(f" - Mean abs error (recomputed): {results_df['abs_error'].mean():.2f}")
print(f" - Median abs error: {results_df['abs_error'].median():.2f}")

# Correlation between true and predicted RUL
corr = np.corrcoef(results_df["true_rul"], results_df["pred_rul"])[0, 1]
print(f" - Pearson correlation (true vs pred): {corr:.4f}")

print(" - Sample (first 5 engines):")
print(results_df.head().to_string(index=False))

# ==================================================
# EVALUATION METRICS (NASA STANDARD)
# ==================================================

rmse = np.sqrt(mean_squared_error(true_rul["true_rul"], pred_rul))
mae = mean_absolute_error(true_rul["true_rul"], pred_rul)

# Coefficient of determination
r2 = r2_score(true_rul["true_rul"], pred_rul)

print("\n========== NASA FD001 TEST EVALUATION ==========")
print(f"RMSE (cycles): {rmse:.2f}")
print(f"MAE  (cycles): {mae:.2f}")
print(f"R^2  : {r2:.4f}")
print("===============================================")
