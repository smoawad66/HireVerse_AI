import os
import requests
from TechnicalSkills.helpers import convert_video_to_audio



SPEECH_RECOGNITION_HOST = "advanced-speech-to-text-fast-accurate-and-ai-powered.p.rapidapi.com"
SPEECH_RECOGNITION_API_KEY = "a7706886c9mshca04b4e277c7245p1ba4b3jsn8a50361f0d73"



def recognize_speech(video_path: str):

    id = video_path.split('-')[-1].split('.')[0]

    audio_path = convert_video_to_audio(video_path, id)

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
                        os.remove(audio_path)
                        print('Audio is transcribed and file is removed')

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

    return transcribed_text.strip()

# if __name__ == '__main__':
#     text = recognize_speech("video.avi", 1)
#     print(text)

