import os, csv
import botocore.exceptions
from dotenv import load_dotenv
import os, boto3, time, botocore


BASE_DIR = os.path.dirname(__file__)
METRICS_DIR = os.path.join(BASE_DIR, 'soft_skills/analysis_metrics')


load_dotenv()
BUCKET_NAME = 'myawshierbucket'
s3 = boto3.client('s3')

def download_file(key, local_path):
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Error: File not found in S3: {key}")
        else:
            print(f"Error downloading file: {e}")
        return False
    print('File downloaded')
    return True



def upload_file(local_path, key):
    try:
        s3.upload_file(local_path, BUCKET_NAME, key)
    except botocore.exceptions.ClientError as e:
        print(f"Error uploading file: {e}")
        return False
    return True




def download_applicant_answers(keys): 
    videos_folder = create_folder('technical_skills/videos')

    paths = []
    for key in keys:
        file_name = key.split('/')[-1]
        path = os.path.join(videos_folder, file_name)
        res = download_file(key, path)
        paths.append(None if res is False else path)
    return paths


def download_questions_file(key):
    questions_folder = create_folder('technical_skills/questions')
    
    path = os.path.join(questions_folder, f'questions-{int(time.time())}.json')
    download_file(key, path)
    return path


def setup_logging(interview_id):
    gaze_folder = create_folder(f'{METRICS_DIR}/gaze')

    gaze_csv_path = f'{gaze_folder}/interview-{interview_id}.csv'
    with open(gaze_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Frame', 'FPS', 'Distance_cm', 'Gaze_Direction','Smoothed_EAR_Avg', 'Smoothed_Ratio_H_Avg', 'Smoothed_Ratio_V_Avg', 'Analysis_State', 'Reference_IOD_px', 'Scale_Factor', 'Smiling'])

    head_folder = create_folder(f'{METRICS_DIR}/head')
    head_csv_path = f'{head_folder}/interview-{interview_id}.csv'
    with open(head_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'posture_score', 'posture_confidence', 'posture_feedback', 'head_pose', 'head_pose_confidence', 'overall_confidence', 'fps'])
    
    return gaze_csv_path, head_csv_path




def create_folder(folder_path):
    folder_path = os.path.join(BASE_DIR, folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


