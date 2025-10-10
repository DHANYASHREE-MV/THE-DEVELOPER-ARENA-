# churn_project.py
# ================
# Advanced classifier for churn prediction using Random Forest & XGBoost
# Includes preprocessing, evaluation, and model saving.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

# ====== CONFIG ======
DATA_PATH = "C://Users//Radha//Downloads//churn.csv"   # change to your CSV file
TARGET_COL = "Churn"      # target column must be binary (0/1 or Yes/No)
OUTPUT_DIR = Path("client_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ====== LOAD DATA ======
def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns}")
    
    y = df[TARGET_COL]
    # Convert Yes/No to 1/0 if needed
    if y.dtype == object:
        y = y.map({"Yes":1, "No":0, "yes":1, "no":0}).fillna(y)
    
    X = df.drop(columns=[TARGET_COL])
    return X, y

# ====== PREPROCESSOR ======
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64","float64","bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre

# ====== EVALUATION ======
def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUTPUT_DIR / f"{name}_cm.png", dpi=120, bbox_inches="tight")
    plt.close()

    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title(f"{name} — ROC Curve")
        plt.savefig(OUTPUT_DIR / f"{name}_roc.png", dpi=120, bbox_inches="tight")
        plt.close()

# ====== MAIN PIPELINE ======
def main():
    X, y = load_data()
    preprocessor = build_preprocessor(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Random Forest baseline
    rf = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "RandomForest")
    joblib.dump(rf, OUTPUT_DIR / "rf_churn_model.joblib")

    # XGBoost tuned
    if HAS_XGB:
        xgb = Pipeline([
            ("pre", preprocessor),
            ("xgb", XGBClassifier(
                n_estimators=600,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss"
            ))
        ])
        xgb.fit(X_train, y_train)
        evaluate_model(xgb, X_test, y_test, "XGBoost")
        joblib.dump(xgb, OUTPUT_DIR / "xgb_churn_model.joblib")

    print("\n✅ Models trained and saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
