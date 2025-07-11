from flask import Blueprint, jsonify
from .services import evaluate_cvs

cv_bp = Blueprint('cv_filtration', __name__)

@cv_bp.route('/cv-filtration', methods=['GET'])
def filter_cvs():
    # try:
    scores = evaluate_cvs()
    return jsonify({"cvScores": scores}), 200

    # except ValueError as e:
    #     return jsonify({"error": str(e)}), 422
    # except Exception as e:
    #     return jsonify({"error": "Internal server error"}), 500

