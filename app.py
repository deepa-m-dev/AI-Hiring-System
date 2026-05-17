from flask import Flask, render_template, request
import joblib
import numpy as np
import sqlite3

app = Flask(__name__)

# ==========================================
# DATABASE SETUP
# ==========================================

conn = sqlite3.connect(
    'database/candidates.db',
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS candidates (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    age REAL,
    cgpa REAL,
    experience_years REAL,
    skills_score REAL,

    prediction TEXT,
    confidence REAL
)
''')

conn.commit()

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load('model/hiring_model.pkl')

# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

# ==========================================
# PREDICT PAGE
# ==========================================

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ==========================================
        # GET FORM DATA
        # ==========================================

        age = float(request.form['age'])
        education_level = float(request.form['education_level'])
        university_tier = float(request.form['university_tier'])
        cgpa = float(request.form['cgpa'])
        internships = float(request.form['internships'])
        projects = float(request.form['projects'])
        programming_languages = float(request.form['programming_languages'])
        certifications = float(request.form['certifications'])
        experience_years = float(request.form['experience_years'])
        hackathons = float(request.form['hackathons'])
        research_papers = float(request.form['research_papers'])
        skills_score = float(request.form['skills_score'])
        soft_skills_score = float(request.form['soft_skills_score'])
        resume_length_words = float(request.form['resume_length_words'])
        company_type = float(request.form['company_type'])

        # ==========================================
        # CREATE FEATURE ARRAY
        # ==========================================

        features = np.array([[

            age,
            education_level,
            university_tier,
            cgpa,
            internships,
            projects,
            programming_languages,
            certifications,
            experience_years,
            hackathons,
            research_papers,
            skills_score,
            soft_skills_score,
            resume_length_words,
            company_type

        ]])

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        prediction = model.predict(features)

        probability = model.predict_proba(features)

        result = "Hired" if prediction[0] == 1 else "Rejected"

        confidence = round(np.max(probability) * 100, 2)

        # ==========================================
        # CANDIDATE RATING
        # ==========================================

        if confidence >= 85:
            candidate_rating = "Excellent Candidate"

        elif confidence >= 70:
            candidate_rating = "Strong Candidate"

        elif confidence >= 50:
            candidate_rating = "Moderate Candidate"

        else:
            candidate_rating = "High Risk Candidate"

        # ==========================================
        # AI INSIGHTS
        # ==========================================

        insights = []

        # Skills analysis

        if skills_score >= 80:
            insights.append("Strong technical skill set")

        elif skills_score < 50:
            insights.append("Technical skills need improvement")

        # Soft skills

        if soft_skills_score >= 75:
            insights.append("Excellent communication and teamwork")

        elif soft_skills_score < 50:
            insights.append("Soft skills are below expected level")

        # Experience

        if experience_years >= 3:
            insights.append("Good industry experience")

        elif experience_years == 0:
            insights.append("Fresher candidate with no experience")

        # Projects

        if projects >= 5:
            insights.append("Strong hands-on project exposure")

        # Research

        if research_papers >= 2:
            insights.append("Good research background")

        # CGPA

        if cgpa >= 8.5:
            insights.append("Excellent academic performance")

        elif cgpa < 6:
            insights.append("Low academic performance")

        # ==========================================
        # SAVE TO DATABASE
        # ==========================================

        cursor.execute(
            '''
            INSERT INTO candidates (
                age,
                cgpa,
                experience_years,
                skills_score,
                prediction,
                confidence
            )

            VALUES (?, ?, ?, ?, ?, ?)
            ''',

            (
                age,
                cgpa,
                experience_years,
                skills_score,
                result,
                confidence
            )
        )

        conn.commit()

        # ==========================================
        # RETURN RESULT PAGE
        # ==========================================

        return render_template(

            'result.html',

            prediction=result,

            confidence=confidence,

            candidate_rating=candidate_rating,

            insights=insights
        )

    except Exception as e:

        return f"ERROR: {str(e)}"

# ==========================================
# DASHBOARD ROUTE
# ==========================================

@app.route('/dashboard')
def dashboard():

    # ==========================================
    # TOTAL CANDIDATES
    # ==========================================

    cursor.execute(
        "SELECT COUNT(*) FROM candidates"
    )

    total_candidates = cursor.fetchone()[0]

    # ==========================================
    # HIRED CANDIDATES
    # ==========================================

    cursor.execute(
        "SELECT COUNT(*) FROM candidates WHERE prediction='Hired'"
    )

    hired_candidates = cursor.fetchone()[0]

    # ==========================================
    # REJECTED CANDIDATES
    # ==========================================

    cursor.execute(
        "SELECT COUNT(*) FROM candidates WHERE prediction='Rejected'"
    )

    rejected_candidates = cursor.fetchone()[0]

    # ==========================================
    # RECENT CANDIDATES
    # ==========================================

    cursor.execute(
        '''
        SELECT

            age,
            cgpa,
            experience_years,
            skills_score,
            prediction,
            confidence

        FROM candidates

        ORDER BY id DESC

        LIMIT 10
        '''
    )

    recent_candidates = cursor.fetchall()

    # ==========================================
    # RETURN DASHBOARD
    # ==========================================

    return render_template(

        'dashboard.html',

        total_candidates=total_candidates,

        hired_candidates=hired_candidates,

        rejected_candidates=rejected_candidates,

        recent_candidates=recent_candidates
    )

# ==========================================
# RUN FLASK
# ==========================================

if __name__ == '__main__':

    app.run(debug=True)