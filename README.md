# 🩺 Diabetes Prediction using Machine Learning

A Machine Learning web application that predicts whether a person is likely to have diabetes based on medical parameters. The application is developed using **Python**, **Scikit-learn**, and **Streamlit** with an easy-to-use interface for real-time predictions.

---

## 📌 Features

- Predicts the likelihood of diabetes based on user input.
- Interactive web interface built with Streamlit.
- Trained Machine Learning model for accurate predictions.
- Fast and user-friendly application.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Pickle

---

## 📂 Project Structure

```
diseases-detection/
│
├── app.py                  # Flask web app (deployed on Vercel)
├── agent.py                # AI assistant: Groq LLM + tool-calling loop
├── rag.py                  # BM25 retrieval over knowledge/
├── knowledge/              # Health guidance notes used for RAG
├── templates/index.html    # Web page for the Flask app
├── streamlit_app.py        # Streamlit version for local use
├── train_model.py
├── model.pkl, scaler.pkl, impute_medians.pkl, metrics.pkl
├── diabetes.csv
├── requirements.txt            # Flask app (used by Vercel)
├── requirements-streamlit.txt  # Streamlit app + training
├── README.md
└── Diabetes_Prediction_using_Machine_Learning.ipynb
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Nishagupta2212/diseases-detection.git
```

### Move into the project directory

```bash
cd diseases-detection
```

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Run the application

Web app (same as the Vercel deployment), opens at http://127.0.0.1:5000:

```bash
flask --app app run
```

Streamlit version:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

### Retrain the model

```bash
pip install -r requirements-streamlit.txt
python train_model.py
```

Commit the regenerated `.pkl` files so the deployed app uses the new model.

### Deploy on Vercel

Import the GitHub repository in Vercel and deploy with the default settings.
Vercel detects the Flask `app` in `app.py` and installs `requirements.txt`.

---

## 🤖 AI Assistant (LLM agent + tools + RAG)

The **AI Assistant** tab is a chatbot powered by `qwen/qwen3.8-27b` on [Groq](https://console.groq.com).
It works as an **agent**: the LLM decides which tools to call, the app runs them, and the LLM
writes the answer from the results.

| Tool | What it does |
|---|---|
| `predict_diabetes_risk` | Runs the trained model (e.g. "what if my BMI was 25?") |
| `compare_to_dataset` | Percentile of a value among the 768 patients |
| `get_dataset_statistics` | Mean, quartiles and per-diagnosis averages for a feature |
| `get_model_info` | Accuracy and feature weights |
| `search_health_guidelines` | **RAG**: BM25 search over the guidance notes in `knowledge/` (WHO, ADA, CDC, AHA) |

Answers drawn from the knowledge library show their sources under the reply. To add knowledge,
add a Markdown file to `knowledge/` with a `# Title`, a `Source: name | url` line and `## Section`
blocks; it is indexed automatically when the app starts.

**Setup:** put your Groq key in a `.env` file (`GROQ_API_KEY=...`, see `.env.example`) for local
use, and add `GROQ_API_KEY` under Vercel → Settings → Environment Variables for the deployed app.

---

## 📊 Dataset

The model is trained on the **Pima Indians Diabetes Dataset**. It uses the following medical attributes:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age


---

## 🤖 Machine Learning Workflow

- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training
- Model Evaluation
- Diabetes Prediction

---
## 📊 Results

- **Algorithm:** Support Vector Machine (linear kernel)
- **Training accuracy:** 78.66%
- **Test accuracy:** 77.27%
- **Dataset:** Pima Indians Diabetes Dataset (768 rows × 9 columns)
- **Features used:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
  Diabetes Pedigree Function, Age

---
## 📈 Future Improvements

- Improve prediction accuracy using advanced algorithms.
- Deploy the application on Streamlit Cloud.
- Add data visualization dashboards.
- Support multiple datasets and enhanced input validation.

---
## 📸 App Preview

**Prediction Interface**
<img width="1901" height="901" alt="Screenshot 2026-07-28 175856" src="https://github.com/user-attachments/assets/f2026af1-521d-4562-9d59-d6b47e3bdaa9" />

**Dataset Preview**
<img width="1916" height="903" alt="Screenshot 2026-07-28 175912" src="https://github.com/user-attachments/assets/329d94ac-63eb-49ba-b5e5-52bca55228f9" />

**Model Performance**
<img width="1816" height="742" alt="Screenshot 2026-07-28 175924" src="https://github.com/user-attachments/assets/419e1bad-fd5d-4f86-b82c-99490e6c97ca" />

## 👩‍💻 Author

**Nisha Kumari**

- GitHub: https://github.com/Nishagupta2212
- LinkedIn: - LinkedIn: https://www.linkedin.com/in/nisha-kumari22/

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
