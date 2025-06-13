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

    technical_results = evaluate_technical_skills(local_questions_path, local_answers_paths)
    evaluate_soft_skills(local_answers_paths, interview_id)

    print(f"Interview {interview_id} analysis finished!")

    requests.post(callback_url, json={
        "results": {
            "technicalSkills": technical_results,
            "softSkills": 'wait'
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




