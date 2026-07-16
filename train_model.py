"""Trains the SVM diabetes classifier and saves the model + scaler to disk."""
import pickle

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes_dataset = pd.read_csv("diabetes.csv")

X = diabetes_dataset.drop(columns="Outcome")
Y = diabetes_dataset["Outcome"]

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("metrics.pkl", "wb") as f:
    pickle.dump({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}, f)

print("Saved model.pkl, scaler.pkl, metrics.pkl")
