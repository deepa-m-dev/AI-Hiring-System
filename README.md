# AI Hiring & Recruitment Intelligence System

An end-to-end AI-powered recruitment analytics platform built using Machine Learning, Flask, XGBoost, SQLite, and interactive analytics dashboards.

The system predicts whether a candidate is likely to be hired based on academic, technical, and professional attributes while also generating intelligent candidate insights and recruitment analytics.

---

# 🚀 Features

## ✅ AI Hiring Prediction

* Predicts whether a candidate is likely to be hired
* Uses Machine Learning classification models
* Final deployed model: XGBoost

---

## ✅ Candidate Intelligence Engine

Generates AI-based insights such as:

* Technical skill strength
* Soft skill evaluation
* Academic performance analysis
* Experience evaluation
* Research/project strength

---

## ✅ Recruitment Analytics Dashboard

Interactive dashboard displaying:

* Total candidates analyzed
* Hired vs rejected candidates
* Confidence scores
* Recent predictions
* Candidate analytics

---

## ✅ Database Integration

* Stores candidate prediction history
* Uses SQLite database
* Tracks predictions and confidence scores

---

## ✅ End-to-End ML Pipeline

Includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Model evaluation
* Deployment

---

# 🧠 Machine Learning Workflow

```text id="9m2vka"
Dataset
   ↓
Data Cleaning
   ↓
EDA & Visualization
   ↓
Feature Engineering
   ↓
SMOTE Class Balancing
   ↓
Model Training
   ↓
XGBoost Classifier
   ↓
Prediction System
   ↓
Flask Deployment
```

---

# 📊 Technologies Used

## Backend

* Python
* Flask
* SQLite

---

## Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

---

## Data Visualization

* Matplotlib
* Seaborn

---

## Frontend

* HTML
* CSS
* Bootstrap
* Font Awesome

---

## Tools

* Git
* GitHub

---

# 📈 Model Performance

| Metric              | Value                 |
| ------------------- | --------------------- |
| Model               | XGBoost               |
| Accuracy            | 67%                   |
| Imbalance Handling  | SMOTE                 |
| Classification Type | Binary Classification |

---

# 📂 Project Structure

```text id="6q4wzr"
AI-Hiring-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── hiring.csv
│
├── model/
│   └── hiring_model.pkl
│
├── database/
│   └── candidates.db
│
├── static/
│   ├── css/
│   ├── images/
│   └── js/
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── dashboard.html
│
└── ml/
    └── train_model.py
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash id="5r8kpa"
git clone https://github.com/deepa-m-dev/AI-Hiring-System.git
```

---

## 2️⃣ Open Project

```bash id="8x1mqt"
cd AI-Hiring-System
```

---

## 3️⃣ Create Virtual Environment

```bash id="2v7qrm"
python -m venv .venv
```

---

## 4️⃣ Activate Environment

### Windows

```bash id="7p3xka"
.venv\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```bash id="1m9qvt"
pip install -r requirements.txt
```

---

## 6️⃣ Run Flask App

```bash id="0x4pzn"
python app.py
```

---

# 🌐 Application Modules

| Module            | Description                  |
| ----------------- | ---------------------------- |
| Home Page         | Landing page with analytics  |
| Prediction Module | Candidate evaluation form    |
| AI Result Engine  | Hiring prediction + insights |
| Dashboard         | Recruitment analytics        |
| Database Module   | Candidate history storage    |

---

# 📷 Screenshots

<img width="1920" height="901" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/93f33c13-1178-469e-9d9e-8afafa45ccbb" />


---

# 🔥 Future Improvements

* REST API integration
* Resume PDF Analyzer
* Candidate ranking system
* Export predictions to CSV
* Advanced analytics charts
* Cloud deployment
* Authentication system

---

# 🎯 Learning Outcomes

This project helped in learning:

* Machine Learning deployment
* Flask backend development
* Model evaluation techniques
* Data preprocessing
* SMOTE imbalance handling
* Dashboard creation
* Database integration
* End-to-end AI application development

---

# 👩‍💻 Author

## Deepa M

AI/ML Developer Aspirant
Passionate about building intelligent real-world applications using Machine Learning and Full Stack Development.
