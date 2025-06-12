from flask import Blueprint, jsonify, request
from .services import recommend

rec_bp = Blueprint('recommendation', __name__)

@rec_bp.route('/recommendation', methods=['GET'])
def recommend_jobs():
     data = request.get_json()
     applicants = data['applicants']
     jobs = data['jobs']

     return jsonify({'recommendations': recommend(applicants, jobs)})
