import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

cols = ['engine_id', 'cycle',
        'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

# Resolve data path relative to this script so the script works from any CWD
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / 'data' / 'train_FD001.txt'

df = pd.read_csv(
    data_path,
    sep=r'\s+',
    header=None,
    names=cols
)

df = df.dropna(axis=1)

print(f"Loaded {len(df)} rows from training data")


# --------------------------------------------------
# 2. COMPUTE RUL (GROUND TRUTH)
# --------------------------------------------------

max_cycle_per_engine = df.groupby('engine_id')['cycle'].max()

df['RUL'] = df.apply(
    lambda r: max_cycle_per_engine[r['engine_id']] - r['cycle'],
    axis=1
)

# Normalize RUL (relative health)
df['RUL_RATIO'] = df.apply(
    lambda r: r['RUL'] / max_cycle_per_engine[r['engine_id']],
    axis=1
)

df['RUL_RATIO'] = df['RUL_RATIO'].clip(0, 1)

print("\nRUL_RATIO distribution:")
print(df['RUL_RATIO'].describe())


# --------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------

features = [
    'cycle',
    'op1', 'op2', 'op3',
    's2', 's3', 's4',
    's7', 's11', 's12', 's15'
]

X = df[features]
y = df['RUL_RATIO']
groups = df['engine_id']


# --------------------------------------------------
# 4. ENGINE-WISE TRAIN / VALIDATION SPLIT (IMPORTANT)
# --------------------------------------------------

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


# --------------------------------------------------
# 5. TRAIN RANDOM FOREST
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# --------------------------------------------------
# 6. EVALUATION (RMSE, MAE, R²)
# --------------------------------------------------

train_pred = rf_model.predict(X_train)
val_pred = rf_model.predict(X_val)

# ---- Ratio-space metrics
rmse_train_ratio = np.sqrt(mean_squared_error(y_train, train_pred))
rmse_val_ratio = np.sqrt(mean_squared_error(y_val, val_pred))

mae_train_ratio = mean_absolute_error(y_train, train_pred)
mae_val_ratio = mean_absolute_error(y_val, val_pred)

r2_train = r2_score(y_train, train_pred)
r2_val = r2_score(y_val, val_pred)

# ---- Convert back to RUL (cycles)
train_engine_ids = df.loc[X_train.index, 'engine_id']
val_engine_ids = df.loc[X_val.index, 'engine_id']

train_max_cycles = max_cycle_per_engine.loc[train_engine_ids].values
val_max_cycles = max_cycle_per_engine.loc[val_engine_ids].values

train_pred_rul = train_pred * train_max_cycles
val_pred_rul = val_pred * val_max_cycles

y_train_rul = df.loc[X_train.index, 'RUL'].values
y_val_rul = df.loc[X_val.index, 'RUL'].values

rmse_train_rul = np.sqrt(mean_squared_error(y_train_rul, train_pred_rul))
rmse_val_rul = np.sqrt(mean_squared_error(y_val_rul, val_pred_rul))


# --------------------------------------------------
# 7. PRINT RESULTS (THIS IS WHAT YOU REPORT)
# --------------------------------------------------

print("\n================ TRAINING PERFORMANCE ================")
print(f"RMSE (RUL cycles): {rmse_train_rul:.2f}")
print(f"MAE  (ratio):      {mae_train_ratio:.4f}")
print(f"R²   (ratio):      {r2_train:.4f}")

print("\n=============== VALIDATION PERFORMANCE ===============")
print(f"RMSE (RUL cycles): {rmse_val_rul:.2f}")
print(f"MAE  (ratio):      {mae_val_ratio:.4f}")
print(f"R²   (ratio):      {r2_val:.4f}")


# --------------------------------------------------
# 8. SAVE MODEL (WITH METADATA)
# --------------------------------------------------

model_package = {
    'model': rf_model,
    'features': features,
    'target': 'RUL_RATIO',
    'description': 'RandomForest trained on normalized RUL (engine-wise split)',
    'rmse_validation_cycles': rmse_val_rul
}

model_path = script_dir / 'rf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

print("\n✅ Model saved as rf_model.pkl")
