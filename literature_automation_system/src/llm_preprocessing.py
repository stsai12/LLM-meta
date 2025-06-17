"""
This module provides functions for preprocessing literature data using LLMs.
"""
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the questions for LLM extraction
QA_QUESTIONS = {
    "Sample Size": "What is the sample size?",
    "Study Design": "What is the study design?",
    "Outcome Type": "What is the outcome type?", # or "What are the main outcomes?"
    "Intervention/Exposure": "What was the intervention or exposure?"
}

DEFAULT_MODEL_NAME = "distilbert-base-cased-distilled-squad"

def preprocess_with_llm(articles_data: list[dict], model_name: str = DEFAULT_MODEL_NAME) -> list[dict]:
    """
    Augments article data with fields extracted from abstracts using a Hugging Face
    question-answering model.

    Args:
        articles_data (list[dict]): A list of article dictionaries. Each dictionary
                                     is expected to have an "Abstract" key.
        model_name (str): The name of a pre-trained question-answering model
                          from Hugging Face Hub.
                          Defaults to "distilbert-base-cased-distilled-squad".

    Returns:
        list[dict]: The list of article dictionaries, augmented with LLM-extracted
                    fields ("Sample Size", "Study Design", "Outcome Type",
                    "Intervention/Exposure"). Returns original data if model
                    loading fails or other critical errors occur.
    """
    if not articles_data:
        logging.warning("No article data provided for LLM preprocessing.")
        return []

    try:
        logging.info(f"Initializing QA pipeline with model: {model_name}")
        # It's good practice to also load tokenizer explicitly to handle truncation if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        logging.info("QA pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to load QA model '{model_name}'. Error: {e}")
        logging.error("Please ensure you have 'torch' or 'tensorflow' installed (e.g., pip install torch).")
        logging.error("Also, ensure the model name is correct and you have internet access to download it.")
        return articles_data # Return original data

    augmented_articles = []
    for i, article in enumerate(articles_data):
        logging.info(f"Processing article {i+1}/{len(articles_data)}: {article.get('Title', 'N/A')[:50]}...")
        abstract = article.get("Abstract")
        llm_extracted_fields = {key: "Not found" for key in QA_QUESTIONS.keys()}

        if abstract and isinstance(abstract, str) and abstract.strip():
            for field_key, question in QA_QUESTIONS.items():
                try:
                    # Handle potential truncation for long abstracts
                    # Most QA models have a token limit (e.g., 512 for BERT-like models)
                    # The pipeline usually handles this, but explicit check/truncation can be added if issues arise.
                    # For now, rely on pipeline's default handling.
                    result = qa_pipeline(question=question, context=abstract)
                    if result and result['score'] > 0.1: # Basic confidence threshold
                        llm_extracted_fields[field_key] = result['answer']
                    else:
                        llm_extracted_fields[field_key] = "Not found (low confidence)"
                except Exception as e:
                    logging.warning(f"Error during QA for article (Abstract: {abstract[:100]}...) with question '{question}'. Error: {e}")
                    llm_extracted_fields[field_key] = "Error during processing"
        else:
            logging.info(f"No abstract found or abstract is empty for article: {article.get('Title', 'N/A')[:50]}. Skipping LLM extraction for it.")
            # llm_extracted_fields will remain "Not found" for all keys

        article.update(llm_extracted_fields)
        augmented_articles.append(article)

    logging.info("LLM preprocessing completed.")
    return augmented_articles

if __name__ == '__main__':
    # Example Usage (requires internet for model download on first run, and torch)
    # Ensure you have transformers and torch/tensorflow installed:
    # pip install transformers torch (or tensorflow)

    sample_articles_for_llm = [
        {
            "Title": "Study on Treatment X",
            "Abstract": "This randomized controlled trial investigated Treatment X in 100 patients. The primary outcome was symptom reduction. The study design involved two groups. Treatment X was administered daily."
        },
        {
            "Title": "Review of AI in Medicine",
            "Abstract": "Artificial intelligence is rapidly evolving in medicine. This article reviews current applications. No sample size is applicable as this is a review paper." # LLM might get confused here
        },
        {
            "Title": "Observational Study on Diet Y",
            "Abstract": "We observed 500 participants on Diet Y. This was a prospective cohort study. The main outcome measured was weight change. Exposure was daily consumption of Diet Y."
        },
        {
            "Title": "Paper without Abstract",
            "Abstract": None
        }
    ]

    print("--- Testing LLM preprocessing ---")
    # Using the default model: distilbert-base-cased-distilled-squad
    # You can specify another model like: "bert-large-uncased-whole-word-masking-finetuned-squad"
    # or a medical specific one if available and compatible.
    augmented_results = preprocess_with_llm(sample_articles_for_llm)

    print("\n--- Augmented Articles ---")
    for article in augmented_results:
        print(f"\nTitle: {article['Title']}")
        print(f"  Abstract: {article.get('Abstract', 'N/A')[:60]}...")
        print(f"  Sample Size: {article.get('Sample Size', 'N/A')}")
        print(f"  Study Design: {article.get('Study Design', 'N/A')}")
        print(f"  Outcome Type: {article.get('Outcome Type', 'N/A')}")
        print(f"  Intervention/Exposure: {article.get('Intervention/Exposure', 'N/A')}")

    # Example with a non-existent model to show error handling
    print("\n--- Testing with a non-existent model (expecting error message) ---")
    err_results = preprocess_with_llm(sample_articles_for_llm, model_name="non-existent-model-12345")
    # Check if original data is returned
    assert len(err_results) == len(sample_articles_for_llm)
    assert "Sample Size" not in err_results[0] # Assuming it wasn't there before

    print("\nLLM preprocessing example finished.")
