import pytest
from unittest.mock import patch, MagicMock
import sys
import os
from typing import List, Dict, Any, Optional, Tuple # Added typing
from pytest_mock import MockerFixture # For mocker type hint

# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_preprocessing import preprocess_with_llm, QA_QUESTIONS, DEFAULT_MODEL_NAME

SAMPLE_ARTICLES_FOR_LLM: List[Dict[str, Any]] = [
    {
        "Title": "Study Alpha",
        "Abstract": "This study involved 500 participants. The design was a randomized controlled trial. Outcome was measured by X. Intervention was drug Y."
    },
    {
        "Title": "Study Beta",
        "Abstract": "A cohort study with 1000 subjects. We looked at exposure Z. The primary outcome was event K."
    },
    {
        "Title": "Study Gamma - No Abstract",
        "Abstract": None
    },
    {
        "Title": "Study Delta - Empty Abstract",
        "Abstract": "   "
    }
]

# Mock answers from the QA pipeline
MOCK_QA_ANSWERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Study Alpha": {
        QA_QUESTIONS["Sample Size"]: {"answer": "500 participants", "score": 0.9},
        QA_QUESTIONS["Study Design"]: {"answer": "randomized controlled trial", "score": 0.85},
        QA_QUESTIONS["Outcome Type"]: {"answer": "measured by X", "score": 0.7},
        QA_QUESTIONS["Intervention/Exposure"]: {"answer": "drug Y", "score": 0.92},
    },
    "Study Beta": {
        QA_QUESTIONS["Sample Size"]: {"answer": "1000 subjects", "score": 0.88},
        QA_QUESTIONS["Study Design"]: {"answer": "cohort study", "score": 0.91},
        QA_QUESTIONS["Outcome Type"]: {"answer": "event K", "score": 0.65},
        QA_QUESTIONS["Intervention/Exposure"]: {"answer": "exposure Z", "score": 0.8},
    }
}

@pytest.fixture
def mock_huggingface_pipeline(mocker: MockerFixture) -> Tuple[MagicMock, MagicMock]:
    """Mocks the Hugging Face pipeline object."""
    mock_pipe_instance = MagicMock()

    def side_effect_qa(question: str, context: str) -> Dict[str, Any]:
        article_title_for_context: Optional[str] = None
        for article_data in SAMPLE_ARTICLES_FOR_LLM:
            if article_data["Abstract"] == context:
                article_title_for_context = article_data["Title"]
                break

        if article_title_for_context and article_title_for_context in MOCK_QA_ANSWERS:
            if question in MOCK_QA_ANSWERS[article_title_for_context]:
                return MOCK_QA_ANSWERS[article_title_for_context][question]
        return {"answer": "Mock answer not found for question", "score": 0.05}

    mock_pipe_instance.side_effect = side_effect_qa

    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', return_value=MagicMock())
    mock_pipeline_constructor = mocker.patch('llm_preprocessing.pipeline', return_value=mock_pipe_instance)

    return mock_pipeline_constructor, mock_pipe_instance


def test_preprocess_with_llm_successful_extraction(mock_huggingface_pipeline: Tuple[MagicMock, MagicMock]) -> None:
    """
    Tests LLM preprocessing with successful mock extraction.
    """
    _, mock_pipe_instance = mock_huggingface_pipeline

    articles_to_process: List[Dict[str, Any]] = [SAMPLE_ARTICLES_FOR_LLM[0].copy(), SAMPLE_ARTICLES_FOR_LLM[1].copy()]

    augmented_articles: List[Dict[str, Any]] = preprocess_with_llm(articles_to_process)

    assert len(augmented_articles) == 2

    alpha = augmented_articles[0]
    assert alpha["Sample Size"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Sample Size"]]["answer"]
    assert alpha["Study Design"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Study Design"]]["answer"]
    assert alpha["Outcome Type"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Outcome Type"]]["answer"]
    assert alpha["Intervention/Exposure"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Intervention/Exposure"]]["answer"]

    beta = augmented_articles[1]
    assert beta["Sample Size"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Sample Size"]]["answer"]
    assert beta["Study Design"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Study Design"]]["answer"]
    assert beta["Outcome Type"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Outcome Type"]]["answer"]
    assert beta["Intervention/Exposure"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Intervention/Exposure"]]["answer"]

    expected_calls: int = len(articles_to_process) * len(QA_QUESTIONS)
    assert mock_pipe_instance.call_count == expected_calls


def test_preprocess_with_llm_missing_abstract(mock_huggingface_pipeline: Tuple[MagicMock, MagicMock]) -> None:
    """
    Tests behavior when an abstract is missing or empty.
    """
    _, mock_pipe_instance = mock_huggingface_pipeline

    articles_to_process: List[Dict[str, Any]] = [SAMPLE_ARTICLES_FOR_LLM[2].copy(), SAMPLE_ARTICLES_FOR_LLM[3].copy()]
    augmented_articles: List[Dict[str, Any]] = preprocess_with_llm(articles_to_process)

    assert len(augmented_articles) == 2
    for article in augmented_articles:
        assert article["Sample Size"] == "Not found"
        assert article["Study Design"] == "Not found"
        assert article["Outcome Type"] == "Not found"
        assert article["Intervention/Exposure"] == "Not found"

    mock_pipe_instance.assert_not_called()


def test_preprocess_with_llm_model_loading_failure(mocker: MockerFixture) -> None:
    """
    Tests behavior when the Hugging Face model fails to load.
    """
    mocker.patch('llm_preprocessing.pipeline', side_effect=Exception("Model loading failed"))
    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer loading failed"))
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', side_effect=Exception("Model loading failed"))

    original_articles: List[Dict[str, Any]] = [SAMPLE_ARTICLES_FOR_LLM[0].copy()]

    augmented_articles: List[Dict[str, Any]] = preprocess_with_llm(original_articles, model_name="failing-model")

    assert len(augmented_articles) == 1
    assert "Sample Size" not in augmented_articles[0]
    assert augmented_articles[0]["Title"] == original_articles[0]["Title"]

def test_preprocess_with_llm_empty_input_list() -> None:
    """
    Tests behavior with an empty list of articles.
    """
    augmented_articles: List[Dict[str, Any]] = preprocess_with_llm([])
    assert augmented_articles == []

def test_low_confidence_answer(mocker: MockerFixture) -> None:
    """
    Tests that low confidence answers are marked as "Not found (low confidence)".
    """
    mock_pipe_instance = MagicMock(return_value={"answer": "A very unsure answer", "score": 0.05})
    mocker.patch('llm_preprocessing.pipeline', return_value=mock_pipe_instance)
    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', return_value=MagicMock())

    article_with_abstract: List[Dict[str, Any]] = [SAMPLE_ARTICLES_FOR_LLM[0].copy()]
    augmented_articles: List[Dict[str, Any]] = preprocess_with_llm(article_with_abstract)

    assert augmented_articles[0]["Sample Size"] == "Not found (low confidence)"

def test_qa_pipeline_call_arguments(mock_huggingface_pipeline: Tuple[MagicMock, MagicMock]) -> None:
    """
    Test that the QA pipeline is called with the correct question and context.
    """
    _, mock_pipe_instance = mock_huggingface_pipeline

    article_to_process: List[Dict[str, Any]] = [SAMPLE_ARTICLES_FOR_LLM[0].copy()]

    preprocess_with_llm(article_to_process)

    study_alpha_abstract: Optional[str] = SAMPLE_ARTICLES_FOR_LLM[0]["Abstract"]

    calls = mock_pipe_instance.call_args_list
    expected_questions_asked: set[str] = set()
    for call_args in calls:
        kwargs = call_args.kwargs
        assert kwargs['context'] == study_alpha_abstract
        expected_questions_asked.add(kwargs['question'])

    assert expected_questions_asked == set(QA_QUESTIONS.values())
```
