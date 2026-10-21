"""Trains and selects the best diabetes classifier, saving artifacts to disk."""
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

# In this dataset, 0 encodes a missing measurement for these columns (a
# person cannot have zero glucose or BMI), so it is imputed rather than
# treated as a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]


def zeros_to_nan(X):
    X = X.copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X


def impute(X_fit, *others):
    """Replace missing zeros with medians computed from X_fit only."""
    medians = {col: X_fit.loc[X_fit[col] != 0, col].median() for col in ZERO_AS_MISSING}
    return medians, [X.replace({col: {0: m} for col, m in medians.items()}) for X in (X_fit, *others)]


diabetes_dataset = pd.read_csv("diabetes.csv")

X = diabetes_dataset.drop(columns="Outcome")
Y = diabetes_dataset["Outcome"]

# Step 1: choose the model on a training split and check it on held-out data.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
_, (X_train, X_test) = impute(X_train, X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

candidates = {
    "SVM (linear)": (SVC(kernel="linear"), {"C": [0.1, 1, 10]}),
    "SVM (rbf)": (
        SVC(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=2),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]},
    ),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
}

best_name, best_estimator, best_cv_score = None, None, -1.0
for name, (estimator, param_grid) in candidates.items():
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy")
    search.fit(X_train_scaled, Y_train)
    print(f"{name}: CV accuracy = {search.best_score_:.4f}, params = {search.best_params_}")
    if search.best_score_ > best_cv_score:
        best_name, best_estimator, best_cv_score = name, search.best_estimator_, search.best_score_

test_accuracy = accuracy_score(Y_test, best_estimator.predict(X_test_scaled))

# Step 2: estimate accuracy of the whole procedure (imputation, scaling and the
# chosen model) with repeated cross-validation over all patients. This is far
# more stable than a single 154-patient test split.
pipeline = make_pipeline(
    FunctionTransformer(zeros_to_nan),
    SimpleImputer(strategy="median"),
    StandardScaler(),
    clone(best_estimator),
)
repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2)
cv_scores = cross_val_score(pipeline, X, Y, cv=repeated_cv, scoring="accuracy")

# Step 3: retrain the chosen model on all patients for deployment.
impute_medians, (X_all,) = impute(X)
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
classifier = clone(best_estimator).fit(X_all_scaled, Y)
train_accuracy = accuracy_score(Y, classifier.predict(X_all_scaled))

print(f"\nSelected model: {best_name}")
print(f"Held-out test accuracy: {test_accuracy:.4f}")
print(f"Repeated CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Training accuracy (all {len(X)} patients): {train_accuracy:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("impute_medians.pkl", "wb") as f:
    pickle.dump(impute_medians, f)

with open("metrics.pkl", "wb") as f:
    pickle.dump(
        {
            "model_name": best_name,
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_patients": len(X),
        },
        f,
    )

print("Saved model.pkl, scaler.pkl, impute_medians.pkl, metrics.pkl")
