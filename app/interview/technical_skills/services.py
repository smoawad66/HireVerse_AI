from .helpers import load_questions, recognize_speech, json_numpy_serializer
from .classes import OptimizedAnswerEvaluator
import logging, json, os



evaluator = OptimizedAnswerEvaluator()


def evaluate_technical_skills(questions_path, answers_paths, interview_id):
    
    try:
        questions = load_questions(questions_path)

        if not questions:
            print("No questions loaded. Please create the JSON file with questions.")
            return
        
        
        answers = [recognize_speech(video_path) for video_path in answers_paths]
        
        # answers = [
        #     "is operator is used for checking the identity it checks both variables are pointing to the same object in memory but equal equal operator is used for equality if two variables have the same value",
        #     # """The 'is' operator is for identity, it checks if two variables point to the same object in memory. In contrast, the '==' operator is for equality, meaning it checks if two objects have the same value. For instance, two separate lists with identical contents are equal (== is True) but are not the same object (is is False).""",
        #     # """Python's memory management is done on a private heap. It uses reference counting to track object references. When an object's reference count is zero, it gets deallocated. There's also a garbage collector that handles cyclic references which reference counting alone cannot solve.""",
        # ]


        if len(questions) > len(answers):
            dif = len(questions) - len(answers)
            print(f"Warning: Only {len(answers)} sample answers provided, but {len(questions)} questions loaded for evaluation.")
            answers.extend([''] * dif)

        results = evaluator.evaluate_batch(questions, answers)

        #push file to s3
        output_file_name = os.path.dirname(__file__) + f'/analysis_metrics/interview-{interview_id}.json'
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, default=json_numpy_serializer)
        except Exception as e:
            logging.error(f"Error saving evaluation results: {e}", exc_info=True)
        
        os.remove(questions_path)

        return results

    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}", exc_info=True)




