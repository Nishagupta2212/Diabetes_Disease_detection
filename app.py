"""Flask web app for the PIMA diabetes prediction model (deployable on Vercel)."""
import csv
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# label, unit, min, max, step, help text, (low, high) typical range or None
FIELDS = {
    "Pregnancies": ("Pregnancies", "", 0, 20, 1, "Number of times pregnant.", None),
    "Glucose": (
        "Glucose", "mg/dL", 0, 300, 1,
        "Plasma glucose 2 hours into an oral glucose tolerance test. Enter 0 if unknown.",
        (70, 139),
    ),
    "BloodPressure": (
        "Blood pressure", "mm Hg", 0, 200, 1,
        "Diastolic blood pressure. Enter 0 if unknown.",
        (60, 89),
    ),
    "SkinThickness": (
        "Skin thickness", "mm", 0, 100, 1,
        "Triceps skin fold thickness. Enter 0 if unknown.",
        (10, 40),
    ),
    "Insulin": (
        "Insulin", "µU/mL", 0, 900, 1,
        "2-hour serum insulin. Enter 0 if unknown.",
        (16, 166),
    ),
    "BMI": ("BMI", "kg/m²", 0.0, 70.0, 0.1, "Body mass index. Enter 0 if unknown.", (18.5, 29.9)),
    "DiabetesPedigreeFunction": (
        "Diabetes pedigree", "", 0.0, 3.0, 0.01,
        "Score of diabetes likelihood based on family history.",
        None,
    ),
    "Age": ("Age", "years", 1, 120, 1, "Age in years.", None),
}

PRESETS = {
    "Typical": dict(zip(FEATURES, [1, 120, 70, 20, 80, 32.0, 0.47, 33])),
    "Low risk": dict(zip(FEATURES, [0, 90, 68, 18, 60, 22.5, 0.20, 25])),
    "High risk": dict(zip(FEATURES, [6, 190, 80, 35, 300, 40.0, 1.20, 55])),
}


def load_pickle(name):
    with open(BASE_DIR / name, "rb") as f:
        return pickle.load(f)


model = load_pickle("model.pkl")
scaler = load_pickle("scaler.pkl")
impute_medians = load_pickle("impute_medians.pkl")
metrics = load_pickle("metrics.pkl")

with open(BASE_DIR / "diabetes.csv", newline="") as f:
    reader = csv.reader(f)
    columns = next(reader)
    dataset = np.array([[float(v) for v in row] for row in reader])


def predict(values):
    row = np.array([[values[f] for f in FEATURES]], dtype=float)
    imputed = []
    for col, median in impute_medians.items():
        i = FEATURES.index(col)
        if row[0, i] == 0:
            row[0, i] = median
            imputed.append(col)

    # The scaler was fitted on a DataFrame; pass plain values without feature names.
    std_data = (row - scaler.mean_) / scaler.scale_
    prediction = int(model.predict(std_data)[0])
    probability = float(model.predict_proba(std_data)[0][1]) if hasattr(model, "predict_proba") else None
    return prediction, probability, imputed


def risk_band(prediction, probability):
    if probability is None:
        return ("high", "Likely diabetic") if prediction else ("low", "Likely not diabetic")
    if probability >= 0.6:
        return "high", "High risk"
    if probability >= 0.3:
        return "moderate", "Moderate risk"
    return "low", "Low risk"


def dataset_summary():
    outcome = dataset[:, columns.index("Outcome")]
    by_diagnosis = {
        label: dict(zip(FEATURES, np.round(dataset[outcome == code, :-1].mean(axis=0), 2).tolist()))
        for code, label in [(0, "Non-diabetic"), (1, "Diabetic")]
    }
    describe = {
        col: {
            "count": int(dataset.shape[0]),
            "mean": round(float(dataset[:, i].mean()), 2),
            "std": round(float(dataset[:, i].std(ddof=1)), 2),
            "min": round(float(dataset[:, i].min()), 2),
            "25%": round(float(np.percentile(dataset[:, i], 25)), 2),
            "50%": round(float(np.percentile(dataset[:, i], 50)), 2),
            "75%": round(float(np.percentile(dataset[:, i], 75)), 2),
            "max": round(float(dataset[:, i].max()), 2),
        }
        for i, col in enumerate(columns)
    }
    return {
        "columns": columns,
        "rows": dataset.tolist(),
        "total": int(dataset.shape[0]),
        "diabetic": int(outcome.sum()),
        "by_diagnosis": by_diagnosis,
        "describe": describe,
    }


def model_weights():
    if hasattr(model, "coef_"):
        return model.coef_[0].tolist(), "Effect on diabetes risk (per standard deviation)", True
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist(), "Feature importance", False
    return None, None, False


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


@app.get("/")
def index():
    weights, weights_title, signed = model_weights()
    page_data = {
        "features": FEATURES,
        "fields": {
            f: {"label": l, "unit": u, "min": lo, "max": hi, "step": s, "help": h, "typical": t}
            for f, (l, u, lo, hi, s, h, t) in FIELDS.items()
        },
        "presets": PRESETS,
        "dataset": dataset_summary(),
        "metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()},
        "weights": weights,
        "weights_title": weights_title,
        "weights_signed": signed,
    }
    return render_template("index.html", page_data=page_data)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    values = {}
    for feature in FEATURES:
        _, _, lo, hi, _, _, _ = FIELDS[feature]
        try:
            value = float(payload[feature])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"Missing or invalid value for {FIELDS[feature][0]}."), 400
        if not lo <= value <= hi:
            return jsonify(error=f"{FIELDS[feature][0]} must be between {lo} and {hi}."), 400
        values[feature] = value

    prediction, probability, imputed = predict(values)
    band, headline = risk_band(prediction, probability)
    return jsonify(
        prediction=prediction,
        probability=probability,
        imputed=imputed,
        band=band,
        headline=headline,
        outcome="Likely diabetic" if prediction else "Likely not diabetic",
    )


if __name__ == "__main__":
    app.run(debug=True)
