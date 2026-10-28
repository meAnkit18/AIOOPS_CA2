from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RANDOM_STATE = 42

data = pd.read_csv(DATA_URL)

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
data = data.drop(columns=["customerID"])

X = data.drop(columns=["Churn"])
y = data["Churn"]

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

pipeline = Pipeline(
    [("prep", preprocessor), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
)

search = GridSearchCV(
    pipeline,
    param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    refit=True,
)
search.fit(X_train, y_train)

model = search.best_estimator_

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Best C: {search.best_params_['clf__C']}")
print(f"Cross-validated ROC-AUC: {search.best_score_:.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

artefact = {
    "model": model,
    "params": model.named_steps["clf"].get_params(),
    "threshold": 0.5,
    "feature_names": list(X.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": {
        "cv_roc_auc": round(float(search.best_score_), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    },
    "dataset": {
        "name": "Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)",
        "source_url": DATA_URL,
        "rows": int(len(data)),
    },
}

model_dir = Path("models")

model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(artefact, model_dir / "churn_model.joblib")

print(f"Model saved to {model_dir / 'churn_model.joblib'}")
