"""Trains and selects the best diabetes classifier, saving artifacts to disk."""
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# In this dataset, 0 encodes a missing measurement for these columns (a
# person cannot have zero glucose or BMI), so it is imputed rather than
# treated as a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]

diabetes_dataset = pd.read_csv("diabetes.csv")

X = diabetes_dataset.drop(columns="Outcome")
Y = diabetes_dataset["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
X_train = X_train.copy()
X_test = X_test.copy()

# Medians are computed from the training split only (non-zero entries) to
# avoid leaking information from the test set.
impute_medians = {}
for col in ZERO_AS_MISSING:
    median = X_train.loc[X_train[col] != 0, col].median()
    impute_medians[col] = median
    X_train[col] = X_train[col].replace(0, median)
    X_test[col] = X_test[col].replace(0, median)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
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

classifier = best_estimator
train_accuracy = accuracy_score(Y_train, classifier.predict(X_train_scaled))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test_scaled))

print(f"\nSelected model: {best_name}")
print(f"CV accuracy: {best_cv_score:.4f}")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

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
            "cv_accuracy": best_cv_score,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        },
        f,
    )

print("Saved model.pkl, scaler.pkl, impute_medians.pkl, metrics.pkl")
