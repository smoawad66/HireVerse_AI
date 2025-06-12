from flask import Blueprint, request, jsonify
from .services import evaluate_applicant_answers

tech_bp = Blueprint('technical_skills', __name__)


@tech_bp.route('/answers/evaluation', methods=['GET'])
def evaluate_answers():
    data = request.get_json()
    questions_path = data['questionsPath']
    answers_paths = data['answersPaths']

    print(questions_path)
    print(answers_paths)

    results = evaluate_applicant_answers(questions_path, answers_paths)

    for result in results:
        print(result['scores']['overall'])

    return jsonify({"results": results})