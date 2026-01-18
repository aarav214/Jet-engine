import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

# ==================================================
# LOAD DATA
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_FD001.txt"

COLS = ['engine_id', 'cycle',
        'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLS)
df = df.dropna(axis=1)

# ==================================================
# LABELS
# ==================================================

max_cycle = df.groupby('engine_id')['cycle'].transform('max')
df['RUL'] = max_cycle - df['cycle']
df['RUL_RATIO'] = (df['RUL'] / max_cycle).clip(0, 1)

# ==================================================
# TEMPORAL FEATURES (SAFE)
# ==================================================

WINDOW = 5
SENSORS = ['s2', 's3', 's4', 's7', 's11', 's12', 's15']

for s in SENSORS:
    df[f'{s}_mean'] = (
        df.groupby('engine_id')[s]
          .rolling(WINDOW)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df[f'{s}_std'] = (
        df.groupby('engine_id')[s]
          .rolling(WINDOW)
          .std()
          .reset_index(level=0, drop=True)
    )

df['log_cycle'] = np.log1p(df['cycle'])
df = df.dropna().reset_index(drop=True)

# ==================================================
# FEATURES
# ==================================================

FEATURES = (
    ['log_cycle', 'op1', 'op2', 'op3'] +
    SENSORS +
    [f'{s}_mean' for s in SENSORS] +
    [f'{s}_std' for s in SENSORS]
)

X = df[FEATURES]
y = df['RUL_RATIO']
groups = df['engine_id']

# ==================================================
# SAMPLE WEIGHT (LATE-LIFE FOCUS)
# ==================================================

sample_weight = np.exp(3 * (1 - y.values))

# ==================================================
# MODEL + PARAMETER SPACE (REDUCED BUT STRONG)
# ==================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [400, 600, 800, 1000],
    "max_depth": [20, 24, 28, None],
    "min_samples_leaf": [3, 4, 6],
    "min_samples_split": [6, 8, 10],
    "max_features": [0.5, 0.6, "sqrt"],
    "bootstrap": [True]
}

# ==================================================
# ENGINE-WISE CV
# ==================================================

cv = GroupKFold(n_splits=5)

rmse_scorer = make_scorer(
    lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    greater_is_better=False
)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,              # 🔥 REDUCED
    scoring=rmse_scorer,
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# ==================================================
# RUN SEARCH
# ==================================================

search.fit(X, y, groups=groups, sample_weight=sample_weight)

# ==================================================
# RESULTS
# ==================================================

print("\n===== BEST MODEL FOUND =====")
print("Best RMSE (ratio):", -search.best_score_)
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
