import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load data ──────────────────────────────────────────────────────────────
X_train = pd.read_csv('dataset/processed/chronological/X_train.csv')
X_test  = pd.read_csv('dataset/processed/chronological/X_test.csv')
y_train = pd.read_csv('dataset/processed/chronological/y_train.csv').squeeze()
y_test  = pd.read_csv('dataset/processed/chronological/y_test.csv').squeeze()

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Features         : {list(X_train.columns)}\n")

# ── 2. Evaluation helper ──────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"--- {name} ---")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R²   : {r2:.3f}\n")
    return preds

# ── 3. Linear Regression (baseline) ──────────────────────────────────────────
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", lr, X_test, y_test)

# ── 4. Random Forest ──────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf, X_test, y_test)

# ── 5. XGBoost ────────────────────────────────────────────────────────────────
print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    tree_method='hist'
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
evaluate("XGBoost", xgb, X_test, y_test)

# ── 6. Save all models ────────────────────────────────────────────────────────
models = {
    "linear_regression": lr,
    "random_forest":     rf,
    "xgboost":           xgb,
}

for name, model in models.items():
    os.makedirs("models/chronological", exist_ok=True)
    path = f"models/chronological/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {path}")