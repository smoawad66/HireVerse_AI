from audio_extract import extract_audio
import logging
import json
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
import os, time
import requests


BASE_DIR = os.path.dirname(__file__)
SPEECH_RECOGNITION_HOST = "advanced-speech-to-text-fast-accurate-and-ai-powered.p.rapidapi.com"
SPEECH_RECOGNITION_API_KEY = "a7706886c9mshca04b4e277c7245p1ba4b3jsn8a50361f0d73"
BUCKET_NAME = 'myawshierbucket'



def convert_video_to_audio(video_path):
    audios_folder = os.path.join(BASE_DIR, 'audios')
    if not os.path.exists(audios_folder):
        os.makedirs(audios_folder, exist_ok=True)

    intended_audio_path = os.path.join(audios_folder, f'q-{int(time.time())}.mp3')

    extract_audio(input_path=video_path, output_path=intended_audio_path, overwrite=True)

    if os.path.exists(intended_audio_path):
        print(f"Successfully created MP3 file at '{intended_audio_path}'")
    else:
        print(f"File '{intended_audio_path}' was not created.")

    # os.remove(video_path)

    return intended_audio_path


def recognize_speech(video_path):
    if video_path is None:
        return ''
    
    audio_path = convert_video_to_audio(video_path)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'audio_file': (os.path.basename(audio_path), audio_file)
            }

            headers = {
                "x-rapidapi-key": SPEECH_RECOGNITION_API_KEY,
                "x-rapidapi-host": SPEECH_RECOGNITION_HOST,
            }

            # Make the POST request using the 'files' parameter
            response = requests.post(f'https://{SPEECH_RECOGNITION_HOST}/transcribe', headers=headers, files=files)

            # Check if the request was successful before trying to parse JSON
            if response.status_code == 200:
                print("Request successful!")
                try:
                    json_response = response.json()
                    transcribed_text = None
                    
                    if 'text' in json_response:
                        transcribed_text: str = json_response['text']

                    # --- Store the extracted text ---
                    if transcribed_text is not None:
                        1
                        # output_filename = "transcribed_audio_advanced_api.txt" # Using a distinct filename
                        # with open(output_filename, "w", encoding="utf-8") as f:
                        #     f.write(transcribed_text)
                    else:
                        print("\nCould not find transcription text in the response.")
                        print("Please examine the 'Full JSON Response' above and adjust the code to access the correct key for THIS API.")

                except requests.exceptions.JSONDecodeError:
                    print("Error: Could not decode JSON response. Is the response valid JSON?")
                    print("Response body:", response.text)
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response body:", response.text) # Print raw body for error details

    except requests.exceptions.RequestException as e:
        print(f"A request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    os.remove(audio_path)

    return transcribed_text.strip()


def load_questions(file_path) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if not isinstance(questions, list):
            raise ValueError("JSON file must contain a list of questions")
        
        # os.remove(file_path)

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




_nltk_resources = {}
def lazy_load_nltk_resources():
    """
    Lazy loads and sets up NLTK resources. This function is called only once
    when TextProcessor is initialized.
    """
    if _nltk_resources:  # Already loaded
        return
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        required_data = [
            ('punkt', 'tokenizers/punkt'),
            ('wordnet', 'corpora/wordnet'),
            ('stopwords', 'corpora/stopwords'),
        ]
        for package, path in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                logging.info(f"Downloading NLTK {package}...")
                nltk.download(package, quiet=True)
        _nltk_resources['word_tokenize'] = word_tokenize
        _nltk_resources['sent_tokenize'] = sent_tokenize
        _nltk_resources['lemmatizer'] = WordNetLemmatizer()
        _nltk_resources['stop_words'] = set(stopwords.words('english'))
    except ImportError:
        logging.error("NLTK is required. Install with: pip install nltk")
        raise
    except Exception as e:
        logging.error(f"Error initializing NLTK resources: {e}")
        raise

