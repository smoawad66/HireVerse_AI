from flask_socketio import SocketIO, emit
from flask import Flask, jsonify, request
from CVFiltration.cv_filtration import evaluate_cvs
from SoftSkills.pre_interview import pre_interview_test, save_parameters
from Recommendation.recommend import *
from SoftSkills.helpers import bin2img, init_interview, img2bin, get_default_interview_state
from collections import defaultdict



app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


interview_states = defaultdict(dict)
interview_ids = {}

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in interview_ids:
        interview_id = interview_ids.pop(sid)
        interview_states.pop(interview_id, None)
        print(f'Client disconnected, cleared state for interview_id: {interview_id}')


@socketio.on('start_pre_interview')
def start_pre_interview(data):
    interview_id = data['interview_id']
    interview_ids[request.sid] = interview_id
    
    # Initialize state for this interview
    interview_states[interview_id] = get_default_interview_state()
    emit('phase_started', {'phase': 'distance_monitoring'})

@socketio.on('frame')
def handle_frame(data):
    try:
        if request.sid not in interview_ids:
            emit('error', {'message': 'No interview ID associated with this session'})
            return
            
        interview_id = interview_ids[request.sid] 
        if not interview_id or interview_id not in interview_states:
            emit('error', {'message': 'No interview ID set or state expired'})
            return

        
        frame = bin2img(data['frame'])

        
        result = pre_interview_test(frame, interview_states[interview_id])

        
        if interview_states[interview_id]['user_exit']:
            interview_states.pop(interview_id, None)
            interview_ids.pop(request.sid, None)
            emit('pre_interview_completed', {'calibration_success': False})
            return

        processed_frame = img2bin(result['frame'])
        
        emit('processed_frame', {
            'frame': processed_frame,
            'metrics': result['metrics']
        })

        if result['phase_completed']:
            state = interview_states[interview_id]
            
            if state['phase'] == 'distance_monitoring':
                state['focal_length'] = result['focal_length']
                state['phase'] = 'gaze_calibration'
                emit('phase_started', {'phase': 'gaze_calibration'})
                print(f"focal:  {state['focal_length']}")

            elif state['phase'] == 'gaze_calibration':
                state['dynamic_gaze_thresholds'] = result['dynamic_gaze_thresholds']
                state['reference_iod_store'] = result['reference_iod_store']

                print(f"dynam:  {state['dynamic_gaze_thresholds']}")
                
                save_parameters(interview_id, state['dynamic_gaze_thresholds'], state['reference_iod_store'], state['focal_length'], result['calibration_success'])
                interview_states.pop(interview_id, None)
                interview_ids.pop(request.sid, None)
                emit('pre_interview_completed', {
                    'calibration_success': result['calibration_success']
                })
    except Exception as e:
        print(f"Error in handle_frame: {str(e)}")
        emit('error', {'message': f'Server error: {str(e)}'})


@socketio.on('user_input')
def handle_user_input(data):
    if request.sid not in interview_ids:
        emit('error', {'message': 'No interview ID associated with this session'})
        return
        
    interview_id = interview_ids[request.sid]

    if interview_id in interview_states:
        action = data['action']
        if action == 'proceed':
            interview_states[interview_id]['user_proceed'] = True
        elif action == 'exit':
            interview_states[interview_id]['user_exit'] = True
    else:
        emit('error', {'message': 'No active interview session'})



# HTTP
@app.route('/cv-filtration', methods=['POST'])
def filter_cvs():
    try:
        cv_scores = evaluate_cvs()
        print(cv_scores)
        return jsonify({"cvScores": cv_scores}), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommendation', methods=['GET'])
def recommend_jobs():
    data = request.get_json()
    applicants = data['applicants']
    jobs = data['jobs']

    return jsonify({'recommendations': recommend(applicants, jobs)})



@app.route('/', methods=['GET'])
def test(): return jsonify({'message': 'hello world from Elsayed'})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)