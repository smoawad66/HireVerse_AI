from sklearn.metrics.pairwise import cosine_similarity
from globals import BASE_PATH
import joblib


model = joblib.load(f"{BASE_PATH}/Recommendation/model/model.pkl")
vectorizer = joblib.load(f"{BASE_PATH}/Recommendation/model/vectorizer.pkl")
le = joblib.load(f"{BASE_PATH}/Recommendation/model/label_encoder.pkl")


def recommend(applicants, jobs):
    result = []
    for applicant in applicants:
        applicant_skills = applicant['skills']
        jobs_ids = []
        if applicant_skills:
            recommended = get_recommended_jobs(applicant_skills, jobs)
            jobs_ids = [job['job_id'] for job in recommended]
        result.append({
            'applicantId': applicant['id'],
            'recommendedJobsIds': jobs_ids
        })
    return result


def get_recommended_jobs(applicant_skills, job_list, top_n=5):
    applicant_text = " ".join(applicant_skills)
    applicant_vec = vectorizer.transform([applicant_text])

    job_scores = []
    for job in job_list:
        job_id = job["id"]
        skills = job["skills"]

        job_text = " ".join(skills)
        job_vec = vectorizer.transform([job_text])

        score = cosine_similarity(applicant_vec, job_vec)[0][0]

        job_scores.append({
            "job_id": job_id,
            "score": score,
        })

    ranked = sorted(job_scores, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]
