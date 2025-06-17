"""
Main script for the Literature Automation System.

This script orchestrates the literature search, data structuring, optional LLM preprocessing,
and saving of results. It uses command-line arguments to control its behavior.
"""
import argparse
import logging
import os
import sys

# Add src directory to Python path to allow direct imports
# This assumes main.py is in 'literature_automation_system/' and 'src' is a subdirectory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from literature_search import search_pubmed
    from data_structuring import structure_and_save_data
    from llm_preprocessing import preprocess_with_llm, DEFAULT_MODEL_NAME
    from Bio import Entrez # For setting Entrez.email
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure all modules are in the 'src' directory and dependencies are installed.")
    sys.exit(1)

# Configure logging
# Basic configuration, can be made more sophisticated (e.g., file logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to parse arguments and run the literature automation pipeline.
    """
    parser = argparse.ArgumentParser(description="Literature Automation System")
    parser.add_argument("keywords", type=str, help="Keywords for PubMed search (e.g., 'cancer therapy AI')")
    parser.add_argument("--start_year", type=int, required=True, help="Start year for publication search")
    parser.add_argument("--end_year", type=int, required=True, help="End year for publication search")
    parser.add_argument("--max_results", type=int, default=10, help="Maximum number of results to fetch (default: 10)")
    parser.add_argument("--output_file", type=str, default="literature_results.csv", help="Name of the output CSV file (default: 'literature_results.csv')")
    parser.add_argument("--filter_reviews", action='store_true', help="Flag to filter out reviews/meta-analyses from results")
    parser.add_argument("--enable_llm", action='store_true', help="Flag to enable LLM preprocessing for extracting additional fields")
    parser.add_argument("--llm_model_name", type=str, default=DEFAULT_MODEL_NAME, help=f"Hugging Face model name for LLM preprocessing (default: '{DEFAULT_MODEL_NAME}')")
    parser.add_argument("--entrez_email", type=str, default="your.email@example.com", help="Email address for NCBI Entrez API (required by NCBI)")
    parser.add_argument("--ncbi_api_key", type=str, default=None, help="NCBI API key for Entrez (optional, but recommended for higher request rates)")

    args = parser.parse_args()

    # Set Entrez email globally (required by Bio.Entrez)
    if not args.entrez_email or args.entrez_email == "your.email@example.com":
        logging.warning("Using default Entrez email 'your.email@example.com'. Please provide your own email using --entrez_email for compliance with NCBI guidelines.")
        # Allow script to continue with default for testing, but NCBI might block without a real email.
    Entrez.email = args.entrez_email
    if args.ncbi_api_key:
        Entrez.api_key = args.ncbi_api_key


    logging.info(f"Starting literature search for keywords: '{args.keywords}'")
    logging.info(f"Publication years: {args.start_year}-{args.end_year}, Max results: {args.max_results}")
    if args.filter_reviews:
        logging.info("Review filtering is enabled.")
    if args.enable_llm:
        logging.info(f"LLM preprocessing is enabled with model: {args.llm_model_name}")

    articles = search_pubmed(
        keywords=args.keywords,
        start_year=args.start_year,
        end_year=args.end_year,
        max_results=args.max_results,
        api_key=args.ncbi_api_key # Pass the API key argument directly
    )

    if articles:
        logging.info(f"Found {len(articles)} articles.")

        if args.enable_llm:
            logging.info("Starting LLM preprocessing...")
            articles = preprocess_with_llm(articles, model_name=args.llm_model_name)
            logging.info("LLM preprocessing completed.")

        logging.info(f"Structuring and saving data to '{args.output_file}'...")
        # The structure_and_save_data function saves into a 'data' subdirectory.
        # We pass only the filename.
        saved_filepath = structure_and_save_data(
            articles_data=articles,
            output_filename=args.output_file,
            filter_reviews=args.filter_reviews
        )

        if saved_filepath:
            logging.info(f"Successfully saved results to: {saved_filepath}")
        else:
            logging.error("Failed to save results.")
    else:
        logging.info("No articles found matching your criteria.")

if __name__ == "__main__":
    # Create data directory if it doesn't exist at the root of the project
    # This main.py is inside literature_automation_system, so ../data
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root_dir, "data")
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            logging.info(f"Created data directory: {data_dir}")
        except OSError as e:
            logging.error(f"Failed to create data directory '{data_dir}'. Error: {e}")
            # Decide if script should exit; for now, it will continue,
            # data_structuring module also tries to create it.

    main()
