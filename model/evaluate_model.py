import pickle
from pathlib import Path
import argparse
import sys

print("Starting evaluate_model.py", flush=True)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

# Load model package
script_dir = Path(__file__).resolve().parent
model_path = script_dir / 'rf_model.pkl'
with open(model_path, 'rb') as f:
    pkg = pickle.load(f)

model = pkg['model']
features = pkg['features']

# Load training data (needed for ground-truth RUL)
project_root = script_dir.parent
data_path = project_root / 'data' / 'train_FD001.txt'

# ----------------------------
# Quick evaluation mode (sample engines to limit rows for low-end machines)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Run full evaluation without sampling')
parser.add_argument('--max-rows', type=int, default=5000, help='Max rows to use in quick mode')
args = parser.parse_args()
cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv(data_path, sep=r'\s+', header=None, names=cols)
print(f"Loaded training CSV: {data_path} ({len(df)} rows)", flush=True)

df = df.dropna(axis=1)
# compute RUL
max_cycle_per_engine = df.groupby('engine_id')['cycle'].max()
df['RUL'] = df.apply(lambda r: max_cycle_per_engine[r['engine_id']] - r['cycle'], axis=1)
df['RUL_RATIO'] = df.apply(lambda r: r['RUL'] / max_cycle_per_engine[r['engine_id']], axis=1)
df['RUL_RATIO'] = df['RUL_RATIO'].clip(0, 1)
print("Computed RUL and RUL_RATIO", flush=True)

# If the saved model expects engineered features (rolling means/std or cycle_ratio),
# generate them here to match the training pipeline used in `train_rj.py`.
missing = set(features) - set(df.columns)
if missing:
    # derive sensor base names from requested features (e.g., 's2', 's3', ...)
    sensor_cols = sorted({f for f in features if f.startswith('s') and '_' not in f})
    # rolling window used in training
    WINDOW = 5
    if any(f.endswith('_mean') or f.endswith('_std') for f in missing) or 'cycle_ratio' in missing:
        # ensure max_cycle is available per-row for cycle_ratio and RUL consistency
        max_cycle = df.groupby('engine_id')['cycle'].transform('max')
        # create rolling mean/std for each sensor requested
        for s in sensor_cols:
            if f'{s}_mean' in missing:
                df[f'{s}_mean'] = (
                    df.groupby('engine_id')[s]
                      .rolling(WINDOW)
                      .mean()
                      .reset_index(level=0, drop=True)
                )
            if f'{s}_std' in missing:
                df[f'{s}_std'] = (
                    df.groupby('engine_id')[s]
                      .rolling(WINDOW)
                      .std()
                      .reset_index(level=0, drop=True)
                )
        # lifecycle position
        if 'cycle_ratio' in missing:
            df['cycle_ratio'] = df['cycle'] / max_cycle

        # training script also used a log cycle feature
        if 'log_cycle' in missing:
            df['log_cycle'] = np.log1p(df['cycle'])

        # drop rows with NaN introduced by rolling and reindex to keep indices simple
        df = df.dropna().reset_index(drop=True)
        print(f"After feature gen: {len(df)} rows, columns: {list(df.columns)}", flush=True)

    # re-evaluate missing set (in case generation covered them)
    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"Required feature(s) missing after feature generation: {sorted(missing)}")

# ----------------------------
# Quick sampling to limit runtime on low-end machines
# ----------------------------
if not args.full:
    max_rows = int(args.max_rows)
    sizes = df.groupby('engine_id').size()
    # deterministic sampler
    rng = np.random.RandomState(42)
    engine_ids = sizes.index.tolist()
    rng.shuffle(engine_ids)
    picked = []
    total = 0
    for eid in engine_ids:
        sz = int(sizes.loc[eid])
        if total + sz > max_rows and len(picked) > 0:
            break
        picked.append(eid)
        total += sz
    df = df[df['engine_id'].isin(picked)].reset_index(drop=True)
    print(f"Quick-eval mode: using {len(picked)} engines, {len(df)} rows (max_rows={max_rows})", flush=True)
else:
    print("Full-eval mode: using entire training set", flush=True)

X = df[features]
y = df['RUL_RATIO']
groups = df['engine_id']

# reproduce same group split
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# predictions
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

# ratio metrics
rmse_train_ratio = np.sqrt(mean_squared_error(y_train, train_pred))
rmse_val_ratio = np.sqrt(mean_squared_error(y_val, val_pred))
mae_train_ratio = mean_absolute_error(y_train, train_pred)
mae_val_ratio = mean_absolute_error(y_val, val_pred)
r2_train = r2_score(y_train, train_pred)
r2_val = r2_score(y_val, val_pred)

# convert to RUL cycles
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

print("===== Model Evaluation =====")
print("Train — RMSE (ratio): {:.4f}, RMSE (cycles): {:.2f}, MAE (ratio): {:.4f}, R2: {:.4f}".format(
    rmse_train_ratio, rmse_train_rul, mae_train_ratio, r2_train
))
print("Val   — RMSE (ratio): {:.4f}, RMSE (cycles): {:.2f}, MAE (ratio): {:.4f}, R2: {:.4f}".format(
    rmse_val_ratio, rmse_val_rul, mae_val_ratio, r2_val
))

# update model metadata with validation rmse (cycles) if not present
if isinstance(pkg, dict) and 'rmse_validation_cycles' not in pkg:
    pkg['rmse_validation_cycles'] = float(rmse_val_rul)
    with open(model_path, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"Updated model package with rmse_validation_cycles={rmse_val_rul:.2f}")
else:
    print(f"Model package already contains rmse_validation_cycles={pkg.get('rmse_validation_cycles')}")
