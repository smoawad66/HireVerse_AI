from flask import Blueprint, request, jsonify
from .helpers import download_applicant_answers, download_questions_file
from .technical_skills.services import evaluate_technical_skills
from .soft_skills.services import evaluate_soft_skills, store_calibration_thresholds
import threading, requests


intr_bp = Blueprint('interview', __name__)


# async task
def interview_analysis_task(data):
    interview_id = data['interviewId']
    questions_path = data['questionsPath']
    answers_paths = data['answersPaths']
    callback_url = data['callbackUrl']

    local_questions_path =  download_questions_file(questions_path)
    local_answers_paths = download_applicant_answers(answers_paths)
    #import os
    #lo = os.path.dirname(__file__) + '/technical_skills/videos'
    #k = os.listdir(lo)
    #local_answers_paths = [os.path.join(lo, i) for i in k]

    technical_results = evaluate_technical_skills(local_questions_path, local_answers_paths, interview_id)
    soft_overall_score = evaluate_soft_skills(local_answers_paths, interview_id)

    print(f"Interview {interview_id} analysis finished!")

    requests.post(callback_url, json={
        "results": {
            "technicalSkills": technical_results,
            "softSkills": {"overall": soft_overall_score}
        }
    })


@intr_bp.route('/analysis', methods=['POST'])
def evaluate_answers():
    data = request.get_json()

    background_thread = threading.Thread(
        target=interview_analysis_task,
        args=(data,),
        daemon=True
    )

    background_thread.start()

    return jsonify({"message": "Interview analysis is running!"}), 202



@intr_bp.route('/calibration', methods=['POST'])
def store_calibration():
    data = request.get_json()
    interview_id = data['interview_id']
    thresholds = data['thresholds']

    store_calibration_thresholds(thresholds, interview_id)

    return jsonify({'message': 'Calibration thresholds stored!'})


@intr_bp.route('/<interview_id>/dashboard')
def dashboard_with_id(interview_id):
    """Redirect to the Dash app with interview_id"""
    from flask import redirect
    return redirect(f'/api/interview/dashboard/?id={interview_id}')


