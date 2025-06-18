"""
This module provides functions for structuring and saving literature data.
"""
from typing import List, Dict, Optional
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define review keywords for filtering (case-insensitive)
REVIEW_KEYWORDS = ["review", "meta-analysis", "systematic review"]

def structure_and_save_data(articles_data: List[Dict], output_filename: str, filter_reviews: bool = False) -> Optional[str]:
    """
    Structures article data into a Pandas DataFrame, optionally filters out reviews,
    and saves it to a CSV file in the 'data/' directory.

    Args:
        articles_data (list[dict]): A list of article dictionaries.
        output_filename (str): The name of the CSV file to save the data (e.g., "search_results.csv").
                                The file will be saved in the 'data/' subdirectory.
        filter_reviews (bool): If True, attempts to filter out review articles
                               based on keywords in the title. Defaults to False.

    Returns:
        str | None: The full path to the saved CSV file if successful, otherwise None.
    """
    if not articles_data:
        logging.warning("No article data provided. Nothing to save.")
        return None

    processed_articles = []
    if filter_reviews:
        logging.info("Filtering review articles...")
        for article in articles_data:
            title = article.get("Title", "")
            if title and not any(keyword.lower() in title.lower() for keyword in REVIEW_KEYWORDS):
                processed_articles.append(article)
            elif not title: # Keep articles with no title if any
                processed_articles.append(article)
        logging.info(f"Retained {len(processed_articles)} articles after filtering. Original count: {len(articles_data)}.")
    else:
        processed_articles = articles_data

    if not processed_articles:
        logging.info("No articles remaining after filtering (if applied). Nothing to save.")
        return None

    try:
        df = pd.DataFrame(processed_articles)
        # Ensure standard column order
        columns = ["Title", "Authors", "Journal", "DOI", "Publication Year", "Abstract"]
        # Filter out any columns not present in the DataFrame to avoid KeyError
        df_columns = [col for col in columns if col in df.columns]
        df = df[df_columns]


        # Ensure the 'data' directory exists within 'literature_automation_system'
        # Assuming the script is run from a context where 'literature_automation_system' is accessible
        # or that the path is relative to the project root.
        # For robustness, construct path relative to this file's location or pass project root.
        # For now, let's assume 'data' is a sibling of 'src' or path is handled by caller.

        # Construct the full path for the output file
        # Project root is assumed to be the parent of 'src' where this file is.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

        if not os.path.exists(data_dir):
            logging.info(f"Data directory '{data_dir}' does not exist. Creating it.")
            os.makedirs(data_dir)

        output_filepath = os.path.join(data_dir, output_filename)

        df.to_csv(output_filepath, index=False)
        logging.info(f"Successfully saved {len(df)} articles to '{output_filepath}'")
        return output_filepath

    except Exception as e:
        logging.error(f"Error creating DataFrame or saving to CSV: {e}")
        return None

if __name__ == '__main__':
    # Example Usage
    sample_articles = [
        {
            "Title": "A Great Study on AI", "Authors": ["Author A", "Author B"], "Journal": "Journal of AI",
            "DOI": "10.1000/jai.1", "Publication Year": "2023", "Abstract": "This is a great study..."
        },
        {
            "Title": "Review of Recent AI Advancements", "Authors": ["Reviewer R"], "Journal": "AI Reviews",
            "DOI": "10.1000/air.1", "Publication Year": "2022", "Abstract": "This paper reviews AI..."
        },
        {
            "Title": "Another Important Research", "Authors": ["Author C"], "Journal": "Tech Journal",
            "DOI": "10.1000/tj.2", "Publication Year": "2023", "Abstract": "More research findings..."
        },
         {
            "Title": "A Meta-Analysis of AI Techniques", "Authors": ["Analyst X"], "Journal": "Journal of Meta-Analysis",
            "DOI": "10.1000/jma.1", "Publication Year": "2023", "Abstract": "This meta-analysis covers..."
        }
    ]

    output_csv_filename = "example_output.csv"
    data_dir_for_example = "../data" # Relative to src for the example

    # Create dummy data directory for the example if it doesn't exist
    if not os.path.exists(data_dir_for_example):
        os.makedirs(data_dir_for_example)

    # Test without filtering
    print("\n--- Testing without review filtering ---")
    saved_path_no_filter = structure_and_save_data(sample_articles, output_csv_filename, filter_reviews=False)
    if saved_path_no_filter:
        print(f"Data saved to: {saved_path_no_filter}")
        # You can optionally print the content of the CSV here to verify
        # try:
        #     print(pd.read_csv(saved_path_no_filter))
        # except Exception as e:
        #     print(f"Could not read saved CSV: {e}")


    # Test with filtering
    print("\n--- Testing with review filtering ---")
    output_csv_filtered_filename = "example_output_filtered.csv"
    saved_path_filter = structure_and_save_data(sample_articles, output_csv_filtered_filename, filter_reviews=True)
    if saved_path_filter:
        print(f"Data saved to: {saved_path_filter}")
        # try:
        #     print(pd.read_csv(saved_path_filter))
        # except Exception as e:
        #     print(f"Could not read saved CSV: {e}")

    # Test with empty data
    print("\n--- Testing with empty data ---")
    structure_and_save_data([], "empty_output.csv")

    # Test with data that becomes empty after filtering
    print("\n--- Testing with data that becomes empty after filtering ---")
    review_only_articles = [sample_articles[1], sample_articles[3]]
    structure_and_save_data(review_only_articles, "empty_after_filter.csv", filter_reviews=True)

    # Clean up example files (optional)
    # for fname in [output_csv_filename, output_csv_filtered_filename, "empty_output.csv", "empty_after_filter.csv"]:
    #     full_path = os.path.join(data_dir_for_example, fname)
    #     if os.path.exists(full_path):
    #         os.remove(full_path)
    # if os.path.exists(data_dir_for_example) and not os.listdir(data_dir_for_example):
    #     os.rmdir(data_dir_for_example)
    print("\nExample run finished.")
