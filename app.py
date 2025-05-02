from flask_socketio import SocketIO
from flask import Flask, jsonify
from CVFiltration.cv_filtration import evaluate_cvs
from SoftSkills.soft_skills import test_interview
from SoftSkills.helpers import bin2img, init_interview

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# HTTP
@app.route('/cv-filtration', methods=['POST'])
def filterCV():
    try:
        cv_scores = evaluate_cvs()
        print(cv_scores)
        return jsonify({"cvScores": cv_scores}), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500



interviews = {}

# Websocket
@socketio.on('connect')
def connected(): 
    print("Connected")

@socketio.on('interview-started')
def start_interview(data):
    iid = data['interview_id']
    init_interview(interviews, iid)
    print(f"Interview number {iid} started")


@socketio.on('frame-captured')
def handle_frame(data):
    iid = data['interview_id']
    frame = bin2img(data['binary_image'])

    if iid not in interviews.keys():
        print('Interview hasn\'t started yet!')
    
    posture_history = interviews[iid]
    test_interview(frame, posture_history, iid)


@socketio.on('interview-ended')
def calc_interview_score(data):
    iid = data['interview_id']
    print(f"Interview number {iid} ended, soft skills score is 1%")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)