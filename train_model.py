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

#Hyperparameter tuning for XGBoost
import json
from sklearn.model_selection import RandomizedSearchCV

print("\nStarting hyperparameter tuning for XGBoost...")

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

xgb_base = XGBRegressor(random_state=42, tree_method='hist', verbosity=0)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV R2 : {search.best_score_:.4f}")

tuned_model = search.best_estimator_

#Evaluating the tuned model
tuned_preds = evaluate("XGBoost (Tuned)", tuned_model, X_test, y_test)

# Debug prediction distribution

preds = tuned_model.predict(X_test)

print("\n===== Prediction Statistics =====")

print("Min :", preds.min())

print("Max :", preds.max())

print("Mean:", preds.mean())

print("Std :", preds.std())

print("=================================\n")

tuned_r2   = r2_score(y_test, tuned_preds)
tuned_mae  = mean_absolute_error(y_test, tuned_preds)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))

# Saving the tuned model
os.makedirs("models/chronological", exist_ok=True)

best_model_path = "models/chronological/best_model.pkl"
with open(best_model_path, "wb") as f:
    pickle.dump(tuned_model, f)
print(f"Saved {best_model_path}")

model_info = {
    "best_model_name": "xgboost_tuned",
    "r2_score": round(tuned_r2, 4),
    "mae": round(tuned_mae, 4),
    "rmse": round(tuned_rmse, 4),
    "best_params": search.best_params_,
    "features": list(X_train.columns)  # exact column order predict.py must use
}

model_info_path = "models/chronological/model_info.json"
with open(model_info_path, "w") as f:
    json.dump(model_info, f, indent=2)

print(f"Saved {model_info_path}")
print(f"\nFeatures used: {model_info['features']}")
print(f"Best model: {model_info['best_model_name']} (Test R2 = {model_info['r2_score']})")