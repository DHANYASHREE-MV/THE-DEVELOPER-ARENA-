# week9_hands_on.py
# =================
# Implements Random Forest, XGBoost, and SVM classifiers
# with preprocessing, training, evaluation, and ROC plots.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Try to import xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

# 1) Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Utility function for evaluation
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title(f"{name} — ROC Curve")
        plt.show()

# 2) Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
evaluate_model(rf, "Random Forest")

# 3) XGBoost (if available)
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    evaluate_model(xgb, "XGBoost")

# 4) SVM with scaling
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
])
evaluate_model(svm, "SVM (RBF Kernel)")

# 5) Cross-validation comparison
models = {
    "Random Forest": rf,
    "SVM": svm,
}
if HAS_XGB:
    models["XGBoost"] = xgb

print("\n=== Cross-Validation (Stratified 5-Fold, F1 Score) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
