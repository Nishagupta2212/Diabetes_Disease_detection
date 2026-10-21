"""Streamlit UI for the PIMA diabetes prediction model."""
import pickle

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Check", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("impute_medians.pkl", "rb") as f:
        impute_medians = pickle.load(f)
    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, impute_medians, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv("diabetes.csv")


model, scaler, impute_medians, metrics = load_artifacts()
diabetes_dataset = load_dataset()

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

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {font-family: 'Inter', sans-serif;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}

.hero {display: flex; align-items: center; gap: 1.25rem; padding: 1.75rem 2rem;
       border-radius: 1.25rem; margin-bottom: 1.5rem; color: #fff;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
       box-shadow: 0 12px 32px rgba(79,70,229,0.25);}
.hero-icon {font-size: 2.75rem; background: rgba(255,255,255,0.18); border-radius: 1rem;
            width: 4.5rem; height: 4.5rem; display: flex; align-items: center;
            justify-content: center; flex-shrink: 0;}
.hero-title {font-size: 2rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em;}
.hero-sub {font-size: 1rem; opacity: 0.9; margin-top: 0.35rem;}

.stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
.stTabs [data-baseweb="tab"] {padding: 0.5rem 1.1rem; border-radius: 0.6rem 0.6rem 0 0;
                              font-weight: 600;}

[data-testid="stForm"] {border-radius: 1rem; padding: 1.5rem;
                        border: 1px solid rgba(128,128,128,0.2);
                        box-shadow: 0 4px 20px rgba(0,0,0,0.04);}
.section-label {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
                text-transform: uppercase; color: #7c3aed; margin: 0.75rem 0 0.25rem;}
.card-title {font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;}

.stButton button, .stFormSubmitButton button {border-radius: 0.65rem; font-weight: 600;}
.stFormSubmitButton button {padding: 0.65rem; font-size: 1.05rem;
                            background: linear-gradient(135deg, #4f46e5, #7c3aed);
                            border: none; color: #fff;}
.stFormSubmitButton button:hover {filter: brightness(1.08); color: #fff;}

.placeholder {border: 2px dashed rgba(128,128,128,0.3); border-radius: 1rem;
              padding: 3rem 1.5rem; text-align: center; opacity: 0.8;}
.placeholder-icon {font-size: 2.5rem; margin-bottom: 0.5rem;}

.result-card {border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem; text-align: center;
              border: 1px solid rgba(128,128,128,0.15);}
.result-label {font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
               text-transform: uppercase; opacity: 0.7;}
.result-value {font-size: 3.75rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.03em;}
.result-headline {font-size: 1.1rem; font-weight: 600; margin-top: 0.25rem;}
.meter {height: 10px; border-radius: 999px; background: rgba(128,128,128,0.2);
        margin-top: 1rem; overflow: hidden;}
.meter-fill {height: 100%; border-radius: 999px;}
.risk-low {background: rgba(34,197,94,0.10);}
.risk-low .result-value, .risk-low .result-headline {color: #16a34a;}
.risk-low .meter-fill {background: linear-gradient(90deg, #4ade80, #16a34a);}
.risk-moderate {background: rgba(234,179,8,0.12);}
.risk-moderate .result-value, .risk-moderate .result-headline {color: #ca8a04;}
.risk-moderate .meter-fill {background: linear-gradient(90deg, #facc15, #ca8a04);}
.risk-high {background: rgba(239,68,68,0.10);}
.risk-high .result-value, .risk-high .result-headline {color: #dc2626;}
.risk-high .meter-fill {background: linear-gradient(90deg, #f87171, #dc2626);}

.vitals {display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;}
.vital {border: 1px solid rgba(128,128,128,0.2); border-radius: 0.75rem; padding: 0.7rem 0.85rem;}
.vital-name {font-size: 0.78rem; opacity: 0.7;}
.vital-value {font-size: 1.1rem; font-weight: 700;}
.pill {display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.1rem 0.55rem;
       border-radius: 999px; margin-top: 0.25rem;}
.pill-ok {background: rgba(34,197,94,0.15); color: #16a34a;}
.pill-high {background: rgba(239,68,68,0.15); color: #dc2626;}
.pill-low {background: rgba(59,130,246,0.15); color: #2563eb;}
.pill-none {background: rgba(128,128,128,0.15); opacity: 0.8;}
.note {font-size: 0.8rem; opacity: 0.7; margin-top: 0.75rem;}

[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,0.2); border-radius: 1rem;
                          padding: 1rem 1.25rem;}

[data-testid="stSidebar"] .stButton button {text-align: left; justify-content: flex-start;}
.disclaimer {font-size: 0.8rem; padding: 0.85rem 1rem; border-radius: 0.75rem;
             background: rgba(234,179,8,0.12); border: 1px solid rgba(234,179,8,0.3);}

@media (max-width: 640px) {
  .hero {padding: 1.25rem; flex-direction: column; text-align: center;}
  .hero-title {font-size: 1.5rem;}
  .vitals {grid-template-columns: 1fr;}
}
</style>
""",
    unsafe_allow_html=True,
)


def apply_preset(name):
    for feature, value in PRESETS[name].items():
        st.session_state[feature] = value
    st.session_state.pop("result", None)


if "Glucose" not in st.session_state:
    apply_preset("Typical")


def predict(values):
    input_data = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES, dtype=float)
    imputed = []
    for col, median in impute_medians.items():
        if input_data.loc[0, col] == 0:
            input_data.loc[0, col] = median
            imputed.append(col)

    std_data = scaler.transform(input_data)
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


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk Check")
    st.write(
        "Estimates the likelihood of diabetes from eight routine health measurements."
    )

    st.markdown("#### Try a sample patient")
    st.button("🙂  Typical", on_click=apply_preset, args=("Typical",), width="stretch")
    st.button("💚  Low risk", on_click=apply_preset, args=("Low risk",), width="stretch")
    st.button("❤️  High risk", on_click=apply_preset, args=("High risk",), width="stretch")

    st.write("")
    html(
        '<div class="disclaimer">⚠️ For educational use only. This is not a medical '
        "diagnosis — please consult a healthcare professional.</div>"
    )

html(
    '<div class="hero"><div class="hero-icon">🩺</div><div>'
    '<div class="hero-title">Diabetes Risk Check</div>'
    '<div class="hero-sub">Enter a few routine health measurements to estimate '
    "the likelihood of diabetes.</div></div></div>"
)

tab_predict, tab_data, tab_model = st.tabs(["🔍 Check risk", "📊 Dataset", "🧠 Model"])


def field_input(feature):
    label, unit, lo, hi, step, help_text, _ = FIELDS[feature]
    st.number_input(
        f"{label} ({unit})" if unit else label,
        min_value=lo,
        max_value=hi,
        step=step,
        help=help_text,
        key=feature,
    )


def vital_status(feature, value, imputed):
    typical = FIELDS[feature][6]
    if feature in imputed:
        return "none", "Not provided"
    if typical is None:
        return None, None
    if value > typical[1]:
        return "high", "Above typical"
    if value < typical[0]:
        return "low", "Below typical"
    return "ok", "Typical"


with tab_predict:
    input_col, result_col = st.columns([3, 2], gap="large")

    with input_col:
        with st.form("patient_form"):
            html('<div class="card-title">Patient details</div>')

            html('<div class="section-label">👤 Profile</div>')
            c1, c2 = st.columns(2)
            with c1:
                field_input("Age")
            with c2:
                field_input("Pregnancies")

            html('<div class="section-label">🩸 Blood tests</div>')
            c1, c2 = st.columns(2)
            with c1:
                field_input("Glucose")
            with c2:
                field_input("Insulin")

            html('<div class="section-label">📏 Body measurements</div>')
            c1, c2 = st.columns(2)
            with c1:
                field_input("BMI")
                field_input("SkinThickness")
            with c2:
                field_input("BloodPressure")
                field_input("DiabetesPedigreeFunction")

            st.write("")
            submitted = st.form_submit_button("Check my risk", type="primary", width="stretch")

        if submitted:
            values = {f: st.session_state[f] for f in FEATURES}
            st.session_state["result"] = (values, *predict(values))

    with result_col:
        if "result" not in st.session_state:
            html(
                '<div class="placeholder"><div class="placeholder-icon">📋</div>'
                "<b>Your result will appear here</b><br>"
                "Fill in the details and click <b>Check my risk</b>, "
                "or pick a sample patient from the sidebar.</div>"
            )
        else:
            values, prediction, probability, imputed = st.session_state["result"]
            band, headline = risk_band(prediction, probability)
            outcome = "Likely diabetic" if prediction else "Likely not diabetic"
            if probability is not None:
                html(
                    f'<div class="result-card risk-{band}">'
                    '<div class="result-label">Estimated diabetes risk</div>'
                    f'<div class="result-value">{probability * 100:.0f}%</div>'
                    f'<div class="result-headline">{headline} · {outcome}</div>'
                    f'<div class="meter"><div class="meter-fill" '
                    f'style="width:{probability * 100:.0f}%"></div></div></div>'
                )
            else:
                html(
                    f'<div class="result-card risk-{band}">'
                    '<div class="result-label">Prediction</div>'
                    f'<div class="result-headline">{headline}</div></div>'
                )

            tiles = []
            for feature in FEATURES:
                label, unit = FIELDS[feature][:2]
                value = values[feature]
                status, status_text = vital_status(feature, value, imputed)
                shown = "—" if feature in imputed else f"{value:g} {unit}".strip()
                pill = f'<div class="pill pill-{status}">{status_text}</div>' if status else ""
                tiles.append(
                    f'<div class="vital"><div class="vital-name">{label}</div>'
                    f'<div class="vital-value">{shown}</div>{pill}</div>'
                )
            html(f'<div class="vitals">{"".join(tiles)}</div>')

            notes = ["Typical ranges are general adult reference values, not diagnostic thresholds."]
            if imputed:
                names = ", ".join(FIELDS[f][0] for f in imputed)
                notes.insert(0, f"No value given for {names}; a typical value was used instead.")
            html(f'<div class="note">{"<br>".join(notes)}</div>')

with tab_data:
    total = len(diabetes_dataset)
    diabetic = int(diabetes_dataset["Outcome"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", total)
    c2.metric("Diabetic", f"{diabetic} ({diabetic / total:.0%})")
    c3.metric("Features", len(FEATURES))

    labelled = diabetes_dataset.assign(
        Diagnosis=diabetes_dataset["Outcome"].map({0: "Non-diabetic", 1: "Diabetic"})
    )
    colors = alt.Scale(domain=["Non-diabetic", "Diabetic"], range=["#6366f1", "#ec4899"])

    st.subheader("Feature distribution")
    feature = st.selectbox(
        "Feature", FEATURES, index=1, format_func=lambda f: FIELDS[f][0]
    )
    chart = (
        alt.Chart(labelled)
        .mark_bar(opacity=0.75, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=FIELDS[feature][0]),
            y=alt.Y("count():Q", stack=None, title="Patients"),
            color=alt.Color("Diagnosis:N", scale=colors, title=None),
            tooltip=["Diagnosis:N", "count():Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("Mean feature values by diagnosis"):
        st.dataframe(
            labelled.drop(columns="Outcome").groupby("Diagnosis").mean().round(2),
            width="stretch",
        )
    with st.expander("Raw data"):
        st.dataframe(diabetes_dataset, width="stretch", height=320)
    with st.expander("Statistical summary"):
        st.dataframe(diabetes_dataset.describe().round(2), width="stretch")

with tab_model:
    st.subheader("Model performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expected accuracy",
        f"{metrics['cv_accuracy'] * 100:.1f}%",
        help=(
            f"Mean accuracy over 5-fold cross-validation repeated 10 times on all patients "
            f"(± {metrics['cv_std'] * 100:.1f}% between folds). The most reliable estimate "
            "of how the model performs on new patients."
        ),
    )
    c2.metric(
        "Held-out test accuracy",
        f"{metrics['test_accuracy'] * 100:.1f}%",
        help="Accuracy on one random 20% split of patients, used when choosing the model. "
        "Varies by several points depending on which patients land in the split.",
    )
    c3.metric("Trained on", f"{metrics['n_patients']} patients")
    st.write(f"**Algorithm:** {metrics['model_name']}")

    if hasattr(model, "coef_"):
        weights, title = model.coef_[0], "Effect on diabetes risk (per standard deviation)"
    elif hasattr(model, "feature_importances_"):
        weights, title = model.feature_importances_, "Feature importance"
    else:
        weights = None

    if weights is not None:
        st.subheader("What drives the prediction")
        importance = pd.DataFrame(
            {"Feature": [FIELDS[f][0] for f in FEATURES], "Weight": weights}
        )
        chart = (
            alt.Chart(importance)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Weight:Q", title=title),
                y=alt.Y("Feature:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.Weight > 0, alt.value("#ec4899"), alt.value("#6366f1")
                ),
                tooltip=["Feature:N", alt.Tooltip("Weight:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
        if hasattr(model, "coef_"):
            st.caption("Pink bars increase the predicted risk; purple bars decrease it.")
