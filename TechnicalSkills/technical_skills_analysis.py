import json
import re
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Union
from functools import lru_cache
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import string
from TechnicalSkills.helpers import json_numpy_serializer, load_questions, download_videos
from TechnicalSkills.audio2text import recognize_speech
from globals import BASE_PATH

BERT_MODEL_PATH = f'{BASE_PATH}/CVFiltration/models/bert_model.pkl'
TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Data Classes ---
@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for score thresholds (excellent, good, fair)."""
    excellent: float
    good: float
    fair: float

@dataclass(frozen=True)
class WeightProfile:
    """Weights for different scoring components."""
    keyword: float
    component: float
    criteria: float

    def __post_init__(self):
        """Validate that weights sum approximately to 1.0."""
        total = self.keyword + self.component + self.criteria
        if not (0.95 <= total <= 1.05):  # Allow a small tolerance for floating point arithmetic
            raise ValueError(f"Weights must sum to ~1.0, got {total}")

@dataclass
class EvaluationConfig:
    """Global configuration for the answer evaluation system."""
    difficulty_thresholds: Dict[str, ThresholdConfig] = field(default_factory=lambda: {
        "Easy": ThresholdConfig(excellent=0.8, good=0.65, fair=0.45),
        "Medium": ThresholdConfig(excellent=0.85, good=0.7, fair=0.5),
        "Hard": ThresholdConfig(excellent=0.9, good=0.75, fair=0.55)
    })
    default_weights: WeightProfile = WeightProfile(keyword=1/3, component=1/3, criteria=1/3)
    experience_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "Junior": 0.9, "Mid-level": 1.0, "Senior": 1.1
    })
    similarity_threshold: float = 0.6
    min_answer_length: int = 10  # Minimum number of words for an answer to be considered valid
    model_name: str = TRANSFORMER_MODEL_NAME
    model_cache_path: str = BERT_MODEL_PATH


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



# --- Text Processing ---
class TextProcessor:
    """Handles text preprocessing tasks like tokenization, lemmatization, and stop word removal."""
    def __init__(self):
        lazy_load_nltk_resources()  # Ensure NLTK resources are loaded
        self._word_tokenize = _nltk_resources['word_tokenize']
        self._sent_tokenize = _nltk_resources['sent_tokenize']
        self._lemmatizer = _nltk_resources['lemmatizer']
        self._stop_words = _nltk_resources['stop_words']
        self._whitespace_pattern = re.compile(r'\s+')
        self._punctuation_translation_table = str.maketrans('', '', string.punctuation.replace('-', ''))

    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> Tuple[str, ...]:
        """
        Tokenizes, normalizes, removes punctuation/stopwords, and lemmatizes text.
        """
        if not text or not text.strip():
            return tuple()
        try:
            normalized = self._whitespace_pattern.sub(' ', text).lower().strip()
            normalized = normalized.translate(self._punctuation_translation_table)
            tokens = self._word_tokenize(normalized)
            processed_tokens = [
                self._lemmatizer.lemmatize(word)
                for word in tokens
                if word.isalpha() and len(word) > 1 and word not in self._stop_words
            ]
            return tuple(processed_tokens)
        except Exception as e:
            logging.warning(f"Text preprocessing error for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return tuple(text.lower().split())

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenizes text into a list of sentences."""
        if not text or not text.strip():
            return []
        try:
            return self._sent_tokenize(text)
        except Exception as e:
            logging.warning(f"Sentence tokenization error for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

# --- Model Management ---
class ModelManager:
    """Manages loading and caching of SentenceTransformer models."""
    def __init__(self, model_name: str, cache_path: str):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily loads the SentenceTransformer model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model from cache or downloads it."""
        if self.cache_path.exists():
            try:
                logging.info(f"Loading cached model from {self.cache_path}")
                with open(self.cache_path, 'rb') as f:
                    model = pickle.load(f)
                    
                    if not hasattr(model.tokenizer, 'pad_token') or model.tokenizer.pad_token is None:
                        model.tokenizer.pad_token = '[PAD]'
    
                    if not hasattr(model.tokenizer, 'pad_token_id') or model.tokenizer.pad_token_id is None:
                        model.tokenizer.pad_token_id = 0

                logging.info("Model loaded from cache successfully.")
                return model
            except Exception as e:
                logging.warning(f"Failed to load cached model: {e}. Downloading new model.", exc_info=True)
        logging.info(f"Downloading model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Model cached to {self.cache_path}")
        except Exception as e:
            logging.warning(f"Failed to cache model: {e}", exc_info=True)
        return model

# --- Scoring Engine ---
class ScoringEngine:
    """Calculates various scores for an applicant's answer against expected criteria."""
    def __init__(self, text_processor: TextProcessor, model_manager: ModelManager, config: EvaluationConfig):
        self.text_processor = text_processor
        self.model_manager = model_manager
        self.config = config

    def calculate_keyword_score(self, applicant_tokens: Tuple[str, ...],
                                expected_keywords: List[str]) -> Tuple[List[str], float]:
        """Calculates a keyword match score."""
        if not expected_keywords:
            return [], 1.0
        if not applicant_tokens:
            return [], 0.0
        processed_expected_keywords = {
            self.text_processor._lemmatizer.lemmatize(kw.lower())
            for kw in expected_keywords
        }
        matched_keywords_found = set()
        applicant_token_set = set(applicant_tokens)
        direct_matches = applicant_token_set.intersection(processed_expected_keywords)
        matched_keywords_found.update(direct_matches)
        for expected_kw in processed_expected_keywords:
            if expected_kw in applicant_token_set:
                matched_keywords_found.add(expected_kw)
                continue
            if any(expected_kw in token or token in expected_kw for token in applicant_token_set):
                matched_keywords_found.add(expected_kw)
        score = len(matched_keywords_found) / len(processed_expected_keywords)
        return list(matched_keywords_found), min(score, 1.0)

    def calculate_component_coverage(self, applicant_answer: str,
                                     expected_components: List[str]) -> Tuple[List[str], float]:
        """Calculates coverage of expected components using sentence embeddings."""
        if not expected_components:
            return [], 1.0
        applicant_sentences = self.text_processor.tokenize_sentences(applicant_answer)
        if not applicant_sentences:
            return [], 0.0
        try:
            all_texts = applicant_sentences + expected_components
            embeddings = self.model_manager.model.encode(all_texts, batch_size=32, show_progress_bar=False)
            n_sentences = len(applicant_sentences)
            sentence_embeddings = embeddings[:n_sentences]
            component_embeddings = embeddings[n_sentences:]
            similarities = cosine_similarity(sentence_embeddings.astype(np.float32), component_embeddings.astype(np.float32))
            component_max_similarities = np.max(similarities, axis=0)
            covered_components = [
                comp for comp, score in zip(expected_components, component_max_similarities)
                if score >= self.config.similarity_threshold
            ]
            overall_score = np.mean(component_max_similarities) if component_max_similarities.size > 0 else 0.0
            return covered_components, float(overall_score)
        except Exception as e:
            logging.error(f"Component coverage calculation error: {e}", exc_info=True)
            return [], 0.0

    def calculate_criteria_score(self, applicant_answer: str, assessment_criteria: str) -> float:
        """
        Calculates a score based on how well the answer meets a specific assessment criterion
        using sentence embeddings and cosine similarity.
        """
        if not assessment_criteria or not applicant_answer.strip():
            return 0.0
        try:
            # Encode the applicant's answer and the assessment criterion
            embeddings = self.model_manager.model.encode(
                [applicant_answer, assessment_criteria],
                show_progress_bar=False
            )
            answer_embedding = embeddings[0].reshape(1, -1)
            criteria_embedding = embeddings[1].reshape(1, -1)
            # Calculate cosine similarity
            similarity = cosine_similarity(answer_embedding, criteria_embedding)[0][0]
            return float(np.clip(similarity, 0.0, 1.0))  # Clip score between 0 and 1
        except Exception as e:
            logging.error(f"Criteria score calculation error: {e}", exc_info=True)
            return 0.0

# --- Answer Evaluator ---
class OptimizedAnswerEvaluator:
    """Main class for evaluating technical interview answers."""
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.text_processor = TextProcessor()
        self.model_manager = ModelManager(self.config.model_name, self.config.model_cache_path)
        self.scoring_engine = ScoringEngine(self.text_processor, self.model_manager, self.config)

    def _get_weight_profile(self, question_data: Dict) -> WeightProfile:
        """Returns the weight profile for scoring."""
        return self.config.default_weights

    def _get_adaptive_thresholds(self, question_data: Dict) -> ThresholdConfig:
        """Calculates adaptive scoring thresholds."""
        difficulty = question_data.get("difficulty_level", "Medium")
        experience = question_data.get("experience_level", "Mid-level")
        base_thresholds = self.config.difficulty_thresholds.get(difficulty, self.config.difficulty_thresholds["Medium"])
        modifier = self.config.experience_modifiers.get(experience, 1.0)
        return ThresholdConfig(
            excellent=float(np.clip(base_thresholds.excellent * modifier, 0.3, 0.95)),
            good=float(np.clip(base_thresholds.good * modifier, 0.3, 0.95)),
            fair=float(np.clip(base_thresholds.fair * modifier, 0.3, 0.95))
        )

    def _calculate_weighted_score(self, scores: Dict[str, float], weights: WeightProfile) -> float:
        """Calculates the overall weighted score."""
        return float(scores.get("keyword", 0.0) * weights.keyword +
                     scores.get("component", 0.0) * weights.component +
                     scores.get("criteria", 0.0) * weights.criteria)

    def _get_qualitative_assessment(self, score: float, thresholds: ThresholdConfig) -> str:
        """Determines the qualitative assessment (Excellent, Good, Fair, Poor)."""
        if score >= thresholds.excellent:
            return "Excellent"
        elif score >= thresholds.good:
            return "Good"
        elif score >= thresholds.fair:
            return "Fair"
        else:
            return "Poor"

    def evaluate_answer(self, question_data: Dict, applicant_answer: str) -> Dict:
        """Evaluates a single applicant answer against question criteria."""
        try:
            if not applicant_answer or len(applicant_answer.split()) < self.config.min_answer_length:
                return self._create_error_result(
                    question_data, applicant_answer,
                    f"Answer is too short or empty (min {self.config.min_answer_length} words required)."
                )
            applicant_tokens = self.text_processor.preprocess_text(applicant_answer)
            matched_kw, keyword_score = self.scoring_engine.calculate_keyword_score(
                applicant_tokens, question_data.get("keywords", [])
            )
            covered_comp, component_score = self.scoring_engine.calculate_component_coverage(
                applicant_answer, question_data.get("expected_answer_components", [])
            )
            # Calculate the new criteria score
            criteria_score = self.scoring_engine.calculate_criteria_score(
                applicant_answer, question_data.get("assessment_criteria", "")
            )
            scores = {
                "keyword": float(keyword_score),
                "component": float(component_score),
                "criteria": float(criteria_score)
            }
            weights = self._get_weight_profile(question_data)
            overall_score = self._calculate_weighted_score(scores, weights)
            scores["overall"] = float(overall_score)
            thresholds = self._get_adaptive_thresholds(question_data)
            assessment = self._get_qualitative_assessment(overall_score, thresholds)
            return {
                "question": question_data.get("question", ""),
                "applicant_answer": applicant_answer,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "question_metadata": {
                    "skill": question_data.get("skill", ""),
                    "difficulty": question_data.get("difficulty_level", ""),
                    "experience_level": question_data.get("experience_level", ""),
                    "job_role": question_data.get("job_role", "")
                },
                "matched_keywords": matched_kw,
                "assessment": assessment,
                "thresholds_used": {
                    "excellent": float(thresholds.excellent),
                    "good": float(thresholds.good),
                    "fair": float(thresholds.fair)
                }
            }
        except Exception as e:
            logging.error(f"Evaluation error for question '{question_data.get('Question Text', '')}': {e}", exc_info=True)
            return self._create_error_result(question_data, applicant_answer, str(e))

    def _create_error_result(self, question_data: Dict, answer: str, error: str) -> Dict:
        """Helper to create a consistent error result dictionary."""
        return {
            "question": question_data.get("Question Text", "N/A"),
            "applicant_answer": answer,
            "error": error,
            "scores": {"overall": 0.0},
            "assessment": "Error"
        }

    def evaluate_batch(self, questions: List[Dict], answers: List[str]) -> List[Dict]:
        """Evaluates a batch of answers."""
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
        results = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            logging.info(f"Evaluating answer {i+1}/{len(questions)}...")
            results.append(self.evaluate_answer(question, answer))
        return results



# i want to get the questions file and the answers videos from S3 
def evaluate_applicant_answers(questions_file, answers_paths):
    
    try:
        evaluator = OptimizedAnswerEvaluator()

        questions = load_questions(questions_file)

        if not questions:
            print("No questions loaded. Please create the JSON file with questions.")
            return
        
        
        local_paths = download_videos(answers_paths)

        answers = [recognize_speech(video_path) for video_path in local_paths]
        
        # answers = [
        #     "is operator is used for checking the identity it checks both variables are pointing to the same object in memory but equal equal operator is used for equality if two variables have the same value",
        #     "is operator is used for checking the identity it checks both variables are pointing to the same object in memory but equal equal operator is used for equality if two variables have the same value",
        #     # """The 'is' operator is for identity, it checks if two variables point to the same object in memory. In contrast, the '==' operator is for equality, meaning it checks if two objects have the same value. For instance, two separate lists with identical contents are equal (== is True) but are not the same object (is is False).""",
        #     # """Python's memory management is done on a private heap. It uses reference counting to track object references. When an object's reference count is zero, it gets deallocated. There's also a garbage collector that handles cyclic references which reference counting alone cannot solve.""",
        # ]


        if len(questions) > len(answers):
            print(f"Warning: Only {len(answers)} sample answers provided, but {len(questions)} questions loaded for evaluation.")
            questions = questions[:len(answers)]


        results = evaluator.evaluate_batch(questions, answers)

        return results
        # output_file_name = 'TechnicalSkills/evaluation_results.json'
        # try:
            # with open(output_file_name, 'w', encoding='utf-8') as f:
                # json.dump(batch_results, f, indent=4, default=json_numpy_serializer)
        # except Exception as e:
            # logging.error(f"Error saving evaluation results: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}", exc_info=True)

# if __name__ == '__main__':
#     questions_file = 'sample_questions.json'

#     evaluate_answers(questions_file, 0)