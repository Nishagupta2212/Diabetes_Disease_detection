"""Flask web app for the PIMA diabetes prediction model (deployable on Vercel)."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

import agent
import rag

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


def vital_status(feature, value, imputed):
    typical = FIELDS[feature][6]
    if feature in imputed:
        return "not provided (median used)"
    if typical is None:
        return None
    if value > typical[1]:
        return f"above typical range {typical[0]}-{typical[1]}"
    if value < typical[0]:
        return f"below typical range {typical[0]}-{typical[1]}"
    return f"within typical range {typical[0]}-{typical[1]}"


def validate_values(raw):
    values = {}
    for feature in FEATURES:
        label, _, lo, hi, _, _, _ = FIELDS[feature]
        try:
            value = float(raw[feature])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing or invalid value for {label}.")
        if not lo <= value <= hi:
            raise ValueError(f"{label} must be between {lo} and {hi}.")
        values[feature] = value
    return values


def execute_tool(name, tool_input):
    """Run one of the assistant's tools (see agent.TOOLS)."""
    if name == "predict_diabetes_risk":
        values = validate_values(tool_input)
        prediction, probability, imputed = predict(values)
        band, headline = risk_band(prediction, probability)
        return {
            "probability_of_diabetes": None if probability is None else round(probability, 3),
            "risk_band": headline,
            "predicted_outcome": "diabetic" if prediction else "not diabetic",
            "values_filled_with_median": {f: impute_medians[f] for f in imputed},
            "measurements": {
                f: {"value": values[f], "status": vital_status(f, values[f], imputed)} for f in FEATURES
            },
        }

    if name == "get_dataset_statistics":
        feature = tool_input.get("feature")
        if feature not in columns:
            raise ValueError(f"Unknown feature {feature!r}.")
        summary = dataset_summary()
        stats = {"feature": feature, **summary["describe"][feature]}
        if feature != "Outcome":
            stats["mean_by_diagnosis"] = {d: m[feature] for d, m in summary["by_diagnosis"].items()}
        return stats

    if name == "compare_to_dataset":
        feature, value = tool_input.get("feature"), tool_input.get("value")
        if feature not in FEATURES or not isinstance(value, (int, float)):
            raise ValueError("Give a known feature and a numeric value.")
        col = dataset[:, columns.index(feature)]
        outcome = dataset[:, columns.index("Outcome")]
        at_or_above = outcome[col >= value]

        def percentile(values):
            return round(float((values < value).mean() * 100), 1)

        result = {
            "feature": feature,
            "value": value,
            "percentile_all_patients": percentile(col),
            "percentile_diabetic": percentile(col[outcome == 1]),
            "percentile_non_diabetic": percentile(col[outcome == 0]),
            "patients_at_or_above": int(at_or_above.size),
        }
        if at_or_above.size:
            result["diabetic_share_at_or_above"] = round(float(at_or_above.mean()), 3)
        if feature in impute_medians:
            result["note"] = "0 means 'not measured' for this feature in the dataset."
        return result

    if name == "search_health_guidelines":
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Give a search query.")
        passages = rag.index.search(query, k=4)
        if not passages:
            return {"passages": [], "note": "Nothing in the library matches this query."}
        return {"passages": [{k: p[k] for k in ("title", "source", "url", "text")} for p in passages]}

    if name == "get_model_info":
        weights, weights_title, _ = model_weights()
        info = {k: (round(float(v), 4) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()}
        if weights is not None:
            info["feature_weights"] = {
                "meaning": weights_title,
                "weights": {f: round(w, 3) for f, w in zip(FEATURES, weights)},
            }
        return info

    raise ValueError(f"Unknown tool {name!r}.")


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
    try:
        values = validate_values(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

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


MAX_CHAT_MESSAGES = 20
MAX_MESSAGE_CHARS = 2000


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    history = payload.get("messages")
    if (
        not isinstance(history, list)
        or not history
        or not all(
            isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
            for m in history
        )
        or history[-1]["role"] != "user"
    ):
        return jsonify(error="Send a list of messages ending with a user message."), 400
    if any(len(m["content"]) > MAX_MESSAGE_CHARS for m in history):
        return jsonify(error=f"Messages are limited to {MAX_MESSAGE_CHARS} characters."), 400

    # Keep the most recent messages, starting from a user turn.
    history = history[-MAX_CHAT_MESSAGES:]
    while history[0]["role"] != "user":
        history = history[1:]

    # Give the assistant the values currently in the form, without storing them in the history.
    form = payload.get("form")
    if isinstance(form, dict):
        try:
            values = validate_values(form)
        except ValueError:
            values = None
        if values:
            note = "Values currently entered in the app's form: " + json.dumps(values)
            history[-1] = {"role": "user", "content": f"{note}\n\n{history[-1]['content']}"}

    try:
        reply, tools_used, sources = agent.chat(history, execute_tool)
    except agent.AgentUnavailable as exc:
        return jsonify(error=str(exc)), 503
    return jsonify(reply=reply, tools_used=tools_used, sources=sources)


if __name__ == "__main__":
    app.run(debug=True)
