from audio_extract import extract_audio
import logging
import json
import numpy as np
from typing import List, Dict
from globals import technical_skills_path
from dotenv import load_dotenv
import os, boto3

# logging.basicKConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
BUCKET_NAME = 'ai-interview'
s3 = boto3.client('s3')


def download_videos(keys):
    paths = []
    for key in keys:
        file_name = key.split('/')[-1]
        path = technical_skills_path(f'videos/{file_name}')

        s3.download_file(BUCKET_NAME, key, path)
        paths.append(path)

    return paths



def convert_video_to_audio(video_path, id):
    intended_audio_path = technical_skills_path(f'audios/question-{id}.mp3')

    extract_audio(input_path=video_path, output_path=intended_audio_path, overwrite=True)

    # Verify the file was created
    if os.path.exists(intended_audio_path):
        print(f"Successfully created MP3 file at '{intended_audio_path}'")
    else:
        print(f"File '{intended_audio_path}' was not created.")
        
    return intended_audio_path



def load_questions(file_path) -> List[Dict]:
    """Loads questions from a JSON file."""
    file_path = technical_skills_path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if not isinstance(questions, list):
            raise ValueError("JSON file must contain a list of questions")
        logging.info(f"Loaded {len(questions)} questions from {file_path}")
        return questions
    except FileNotFoundError:
        logging.error(f"Questions file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error loading questions: {e}")
        return []


def json_numpy_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
