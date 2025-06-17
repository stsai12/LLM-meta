import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_preprocessing import preprocess_with_llm, QA_QUESTIONS, DEFAULT_MODEL_NAME

SAMPLE_ARTICLES_FOR_LLM = [
    {
        "Title": "Study Alpha",
        "Abstract": "This study involved 500 participants. The design was a randomized controlled trial. Outcome was measured by X. Intervention was drug Y."
    },
    {
        "Title": "Study Beta",
        "Abstract": "A cohort study with 1000 subjects. We looked at exposure Z. The primary outcome was event K."
        # Missing intervention explicitly, sample size at start.
    },
    {
        "Title": "Study Gamma - No Abstract",
        "Abstract": None
    },
    {
        "Title": "Study Delta - Empty Abstract",
        "Abstract": "   " # Whitespace only
    }
]

# Mock answers from the QA pipeline
MOCK_QA_ANSWERS = {
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
        QA_QUESTIONS["Intervention/Exposure"]: {"answer": "exposure Z", "score": 0.8}, # Model might find this
    }
}

@pytest.fixture
def mock_huggingface_pipeline(mocker):
    """Mocks the Hugging Face pipeline object."""
    mock_pipe_instance = MagicMock()

    def side_effect_qa(question, context):
        # Find which article's abstract is being processed (simple check)
        article_title_for_context = None
        for article_data in SAMPLE_ARTICLES_FOR_LLM:
            if article_data["Abstract"] == context:
                article_title_for_context = article_data["Title"]
                break

        if article_title_for_context and article_title_for_context in MOCK_QA_ANSWERS:
            if question in MOCK_QA_ANSWERS[article_title_for_context]:
                return MOCK_QA_ANSWERS[article_title_for_context][question]
        return {"answer": "Mock answer not found for question", "score": 0.05} # Low confidence default

    mock_pipe_instance.side_effect = side_effect_qa

    # Mock the pipeline function call itself to return our mock_pipe_instance
    # Also mock AutoTokenizer and AutoModelForQuestionAnswering to prevent actual loading
    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', return_value=MagicMock())
    mock_pipeline_constructor = mocker.patch('llm_preprocessing.pipeline', return_value=mock_pipe_instance)

    return mock_pipeline_constructor, mock_pipe_instance


def test_preprocess_with_llm_successful_extraction(mock_huggingface_pipeline):
    """
    Tests LLM preprocessing with successful mock extraction.
    """
    _, mock_pipe_instance = mock_huggingface_pipeline # mock_pipeline_constructor also available

    # Process only the first two articles which have abstracts and mock answers
    articles_to_process = [SAMPLE_ARTICLES_FOR_LLM[0], SAMPLE_ARTICLES_FOR_LLM[1]]

    augmented_articles = preprocess_with_llm(articles_to_process.copy()) # Use .copy()

    assert len(augmented_articles) == 2

    # Article 1 ("Study Alpha")
    alpha = augmented_articles[0]
    assert alpha["Sample Size"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Sample Size"]]["answer"]
    assert alpha["Study Design"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Study Design"]]["answer"]
    assert alpha["Outcome Type"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Outcome Type"]]["answer"]
    assert alpha["Intervention/Exposure"] == MOCK_QA_ANSWERS["Study Alpha"][QA_QUESTIONS["Intervention/Exposure"]]["answer"]

    # Article 2 ("Study Beta")
    beta = augmented_articles[1]
    assert beta["Sample Size"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Sample Size"]]["answer"]
    assert beta["Study Design"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Study Design"]]["answer"]
    assert beta["Outcome Type"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Outcome Type"]]["answer"]
    assert beta["Intervention/Exposure"] == MOCK_QA_ANSWERS["Study Beta"][QA_QUESTIONS["Intervention/Exposure"]]["answer"]

    # Check how many times the pipeline was called
    # For each article, it's called for each question in QA_QUESTIONS
    expected_calls = len(articles_to_process) * len(QA_QUESTIONS)
    assert mock_pipe_instance.call_count == expected_calls


def test_preprocess_with_llm_missing_abstract(mock_huggingface_pipeline):
    """
    Tests behavior when an abstract is missing or empty.
    """
    mock_pipeline_constructor, mock_pipe_instance = mock_huggingface_pipeline

    # Articles 2 (no abstract) and 3 (empty abstract)
    articles_to_process = [SAMPLE_ARTICLES_FOR_LLM[2], SAMPLE_ARTICLES_FOR_LLM[3]]
    augmented_articles = preprocess_with_llm(articles_to_process.copy())

    assert len(augmented_articles) == 2
    for article in augmented_articles:
        assert article["Sample Size"] == "Not found"
        assert article["Study Design"] == "Not found"
        assert article["Outcome Type"] == "Not found"
        assert article["Intervention/Exposure"] == "Not found"

    mock_pipe_instance.assert_not_called() # Pipeline should not be called if no abstract


def test_preprocess_with_llm_model_loading_failure(mocker):
    """
    Tests behavior when the Hugging Face model fails to load.
    """
    # Make the pipeline constructor raise an exception
    mocker.patch('llm_preprocessing.pipeline', side_effect=Exception("Model loading failed"))
    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer loading failed"))
    # No need to mock AutoModelForQuestionAnswering if Tokenizer already fails, or mock it too.
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', side_effect=Exception("Model loading failed"))


    original_articles = [SAMPLE_ARTICLES_FOR_LLM[0].copy()] # Process one article

    # Use caplog to capture logging output
    # import logging
    # with caplog.at_level(logging.ERROR):
    augmented_articles = preprocess_with_llm(original_articles, model_name="failing-model")
        # assert "Failed to load QA model" in caplog.text
        # assert "Please ensure you have 'torch' or 'tensorflow' installed" in caplog.text

    # Should return the original data without LLM fields
    assert len(augmented_articles) == 1
    assert "Sample Size" not in augmented_articles[0]
    assert augmented_articles[0]["Title"] == original_articles[0]["Title"] # Ensure it's the same data

def test_preprocess_with_llm_empty_input_list():
    """
    Tests behavior with an empty list of articles.
    """
    augmented_articles = preprocess_with_llm([])
    assert augmented_articles == []

def test_low_confidence_answer(mocker):
    """
    Tests that low confidence answers are marked as "Not found (low confidence)".
    """
    mock_pipe_instance = MagicMock(return_value={"answer": "A very unsure answer", "score": 0.05}) # Low score
    mocker.patch('llm_preprocessing.pipeline', return_value=mock_pipe_instance)
    mocker.patch('llm_preprocessing.AutoTokenizer.from_pretrained', return_value=MagicMock())
    mocker.patch('llm_preprocessing.AutoModelForQuestionAnswering.from_pretrained', return_value=MagicMock())

    article_with_abstract = [SAMPLE_ARTICLES_FOR_LLM[0].copy()]
    augmented_articles = preprocess_with_llm(article_with_abstract)

    assert augmented_articles[0]["Sample Size"] == "Not found (low confidence)"
    # This will apply to all fields as the mock_pipe_instance returns the same for all questions here.

def test_qa_pipeline_call_arguments(mock_huggingface_pipeline):
    """
    Test that the QA pipeline is called with the correct question and context.
    """
    _, mock_pipe_instance = mock_huggingface_pipeline

    article_to_process = [SAMPLE_ARTICLES_FOR_LLM[0].copy()] # "Study Alpha"

    preprocess_with_llm(article_to_process)

    # Check calls for "Study Alpha"
    study_alpha_abstract = SAMPLE_ARTICLES_FOR_LLM[0]["Abstract"]

    calls = mock_pipe_instance.call_args_list
    expected_questions_asked = set()
    for call_args in calls:
        kwargs = call_args.kwargs
        assert kwargs['context'] == study_alpha_abstract
        expected_questions_asked.add(kwargs['question'])

    assert expected_questions_asked == set(QA_QUESTIONS.values())
```
