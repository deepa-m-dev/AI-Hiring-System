import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('../dataset/hiring.csv')

# =========================
# BASIC DATASET ANALYSIS
# =========================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nSTATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

# =========================
# DROP USELESS COLUMN
# =========================

df = df.drop('candidate_id', axis=1)

# =========================
# HIRING DISTRIBUTION
# =========================

sns.countplot(x='hired', data=df)

plt.title("Hiring Distribution")

plt.savefig('../static/images/hiring_distribution.png')

plt.close()

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(14,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.savefig('../static/images/correlation_heatmap.png')

plt.close()

# =========================
# CGPA DISTRIBUTION
# =========================

sns.histplot(df['cgpa'], kde=True)

plt.title("CGPA Distribution")

plt.savefig('../static/images/cgpa_distribution.png')

plt.close()

# =========================
# SKILLS SCORE DISTRIBUTION
# =========================

sns.histplot(df['skills_score'], kde=True)

plt.title("Skills Score Distribution")

plt.savefig('../static/images/skills_score_distribution.png')

plt.close()

# =========================
# EXPERIENCE DISTRIBUTION
# =========================

sns.histplot(df['experience_years'], kde=True)

plt.title("Experience Years Distribution")

plt.savefig('../static/images/experience_distribution.png')

plt.close()

# =========================
# HIRING BY EDUCATION LEVEL
# =========================

plt.figure(figsize=(8,5))

sns.countplot(
    x='education_level',
    hue='hired',
    data=df
)

plt.title("Hiring by Education Level")

plt.savefig('../static/images/hiring_by_education.png')

plt.close()

# =========================
# SKILLS SCORE VS HIRING
# =========================

sns.boxplot(
    x='hired',
    y='skills_score',
    data=df
)

plt.title("Skills Score vs Hiring")

plt.savefig('../static/images/skills_vs_hiring.png')

plt.close()

# =========================
# CGPA VS HIRING
# =========================

sns.boxplot(
    x='hired',
    y='cgpa',
    data=df
)

plt.title("CGPA vs Hiring")

plt.savefig('../static/images/cgpa_vs_hiring.png')

plt.close()

# =========================
# EXPERIENCE VS HIRING
# =========================

sns.boxplot(
    x='hired',
    y='experience_years',
    data=df
)

plt.title("Experience vs Hiring")

plt.savefig('../static/images/experience_vs_hiring.png')

plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("\nCharts saved inside static/images/")


# ==========================================
# MACHINE LEARNING SECTION
# ==========================================



# ==========================================
# HANDLE MISSING VALUES
# ==========================================

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

label_encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

print("\nDATA AFTER ENCODING")
print(df.head())



print("\nLABEL ENCODINGS")

for col, encoder in label_encoders.items():

    print(f"\n{col}")

    for index, label in enumerate(encoder.classes_):
        print(f"{label} --> {index}")

# ==========================================
# FEATURE & TARGET SPLIT
# ==========================================

X = df.drop('hired', axis=1)

y = df['hired']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)
print("TEST DATA SIZE:", X_test.shape)


# ==========================================
# HANDLE CLASS IMBALANCE USING SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAFTER SMOTE")

print(y_train.value_counts())

# ==========================================
# TRAIN MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(accuracy)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig('../static/images/confusion_matrix.png')

plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df)

plt.figure(figsize=(12,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Feature Importance")

plt.savefig('../static/images/feature_importance.png')

plt.close()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, 'hiring_model.pkl')

print("\nMODEL SAVED SUCCESSFULLY!")