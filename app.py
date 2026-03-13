import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import numpy as np
import joblib
import pdfplumber

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# -------- Extract Text From PDF --------
def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text.lower()


# -------- Home Page --------
@app.route('/')
def home():
    return render_template("index.html")


# -------- Resume Upload --------
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['resume']

    text = extract_text_from_pdf(file)

    # -------- Feature Extraction --------

    years_experience = min(text.count("experience") * 2, 15)

    skills = [
        "python",
        "machine learning",
        "data science",
        "ai",
        "deep learning",
        "flask",
        "sql",
        "nlp"
    ]

    skills_match_score = min(sum(text.count(skill) for skill in skills) * 15, 100)

    if "phd" in text:
        education_level = 3
    elif "master" in text:
        education_level = 2
    else:
        education_level = 1

    project_count = min(text.count("project"), 10)

    resume_length = min(len(text), 900)

    github_activity = min(text.count("github") * 80, 800)

    features = [[
        years_experience,
        skills_match_score,
        education_level,
        project_count,
        resume_length,
        github_activity
    ]]

    print("Extracted Features:", features)

    # -------- Prediction --------

    prediction = model.predict(features)

    if prediction[0] == "Yes" or prediction[0] == 1:
        result = "✅ Candidate Shortlisted"
    else:
        result = "❌ Candidate Not Shortlisted"

    # -------- Resume Score --------

    score = (
        (years_experience/15)*25 +
        (skills_match_score/100)*35 +
        (project_count/10)*15 +
        (github_activity/800)*15 +
        (education_level/3)*10
    )

    score = round(score, 2)

    return render_template(
        "index.html",
        prediction_text=result,
        resume_score=score
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)