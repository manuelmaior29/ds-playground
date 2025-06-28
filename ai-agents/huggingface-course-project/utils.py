HF_QUESTIONS_API = "https://agents-course-unit4-scoring.hf.space"

import requests

def get_question(id: int) -> dict:
    """
    Get a question from the Huggingface API.

    Args:
        id (int): The id of the question to get.

    Returns:
        dict: The question as a dictionary.
    """
    response = requests.get(f"{HF_QUESTIONS_API}/questions/")
    questions = response.json()
    question = questions[id]
    return question
    