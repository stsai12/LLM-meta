"""
This module provides functions for searching and retrieving literature from PubMed and Semantic Scholar.
"""
import requests
import time
from typing import List, Dict, Optional
from Bio import Entrez, Medline
import logging

# Configure logging
# Ensure logger is configured. If main.py also configures, this might be redundant
# or could be improved by using a named logger.
logger = logging.getLogger(__name__)
if not logger.handlers: # Check if logger already has handlers
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Semantic Scholar Search Function ---
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_SEMANTIC_SCHOLAR_FIELDS = "title,authors.name,year,abstract,journal,externalIds"

def search_semanticscholar(
    keywords: str,
    start_year: int,
    end_year: int,
    max_results: int = 20
) -> List[Dict]:
    """
    Searches Semantic Scholar for articles based on keywords and a date range.

    Args:
        keywords (str): Search terms (e.g., "CRISPR gene editing").
        start_year (int): Start of the publication year range.
        end_year (int): End of the publication year range.
        max_results (int): Maximum number of articles to fetch. Defaults to 20.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    details of an article. Returns an empty list if
                    an error occurs or no articles are found.
    """
    logger.info(f"Searching Semantic Scholar for: '{keywords}', Year: {start_year}-{end_year}, Max results: {max_results}")
    articles = []
    query_params = {
        "query": keywords,
        "limit": max_results,
        "fields": DEFAULT_SEMANTIC_SCHOLAR_FIELDS,
        "year": f"{start_year}-{end_year}"
    }

    try:
        # Semantic Scholar API has a rate limit of 100 requests per 5 minutes for non-authenticated users.
        # A single search is one request. Adding a small delay as a general good practice.
        time.sleep(1) # Politeness delay

        response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=query_params, timeout=20) # Added timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()

        if not data.get("data"):
            logger.info("No articles found on Semantic Scholar for the given query.")
            return []

        for item in data["data"]:
            # Normalize Semantic Scholar fields to match PubMed output structure
            title = item.get("title")
            # Authors is a list of dicts, extract 'name'
            authors = [author["name"] for author in item.get("authors", []) if author.get("name")]
            journal_info = item.get("journal")
            journal_name = journal_info.get("name") if journal_info else None
            doi = item.get("externalIds", {}).get("DOI")
            pub_year = str(item.get("year")) if item.get("year") else None # Ensure year is string
            abstract = item.get("abstract")

            articles.append({
                "Title": title,
                "Authors": authors,
                "Journal": journal_name,
                "DOI": doi,
                "Publication Year": pub_year,
                "Abstract": abstract,
                "Source": "Semantic Scholar" # Add source information
            })
        logger.info(f"Successfully fetched and parsed {len(articles)} articles from Semantic Scholar.")

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while searching Semantic Scholar: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred while searching Semantic Scholar: {req_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Semantic Scholar search: {e}")

    return articles


# --- PubMed Search Function (remains largely the same) ---
def search_pubmed(keywords: str, start_year: int, end_year: int, max_results: int = 20, api_key: Optional[str] = None) -> List[Dict]:
    """
    Searches PubMed for articles based on keywords and a date range.

    Args:
        keywords (str): Search terms (e.g., "CRISPR gene editing").
        start_year (int): Start of the publication year range.
        end_year (int): End of the publication year range.
        max_results (int): Maximum number of articles to fetch. Defaults to 20.
        api_key (str, optional): NCBI API key for Entrez. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    details of an article (Title, Authors, Journal, DOI,
                    Publication Year, Abstract). Returns an empty list if
                    an error occurs or no articles are found.
    """
    # Entrez.email must be set globally before calling this function.
    # This is typically handled by the main script or a higher-level function.
    if api_key: # This is correctly passed.
        Entrez.api_key = api_key

    search_term = f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])"
    articles = []

    try:
        logger.info(f"Searching PubMed with query: {search_term}, max_results={max_results}")
        # Ensure Entrez.email is set
        if not Entrez.email:
            logger.error("Entrez.email not set. NCBI requires this for API calls.")
            # This ideally should be a configuration error raised earlier,
            # but as a safeguard:
            return []

        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]

        if not pmids:
            logger.info("No articles found on PubMed for the given query.")
            return []

        logger.info(f"Found {len(pmids)} PMIDs from PubMed. Fetching details...")
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
        medline_records = Medline.parse(handle)

        for medline_record in medline_records:
            try:
                title = medline_record.get("TI", None)
                authors = medline_record.get("AU", [])
                journal = medline_record.get("JT", None)
                doi = None
                if "LID" in medline_record:
                    for item in medline_record["LID"]:
                        if "[doi]" in item:
                            doi = item.replace("[doi]", "").strip()
                            break
                if doi is None and "AID" in medline_record:
                     for item in medline_record["AID"]:
                        if "[doi]" in item:
                            doi = item.replace("[doi]", "").strip()
                            break
                pub_year_full = medline_record.get("DP", "")
                pub_year = pub_year_full[:4] if pub_year_full else None
                abstract = medline_record.get("AB", None)

                articles.append({
                    "Title": title,
                    "Authors": authors,
                    "Journal": journal,
                    "DOI": doi,
                    "Publication Year": pub_year,
                    "Abstract": abstract,
                    "Source": "PubMed" # Add source information
                })
            except Exception as e:
                logger.error(f"Error parsing PubMed Medline record: {medline_record.get('PMID', 'Unknown PMID')}. Error: {e}")
                continue

        handle.close()
        logger.info(f"Successfully fetched and parsed {len(articles)} articles from PubMed.")

    except Exception as e:
        logger.error(f"An error occurred during PubMed search or fetch: {e}")
        return []

    return articles


# --- Dispatcher Function ---
def search_literature(
    source: str,
    keywords: str,
    start_year: int,
    end_year: int,
    max_results: int,
    pubmed_api_key: Optional[str] = None
) -> List[Dict]:
    """
    Dispatcher function to search literature from the specified source.

    Args:
        source (str): The literature source ('pubmed', 'semanticscholar', 'all').
        keywords (str): Search terms.
        start_year (int): Start publication year.
        end_year (int): End publication year.
        max_results (int): Maximum number of results to fetch.
                           If 'all', this number is targeted for each source.
        pubmed_api_key (str, optional): NCBI API key for PubMed.

    Returns:
        list[dict]: A list of articles from the specified source(s).
    """
    all_articles = []

    if source == "pubmed":
        # Entrez.email needs to be set before this call.
        # Assuming it's set by the main application.
        if not Entrez.email:
            logger.error("Entrez.email not set. Required for PubMed search.")
            # Consider how to handle this: raise error, or expect it to be set.
            # For now, proceed, but main.py must set it.
        all_articles = search_pubmed(keywords, start_year, end_year, max_results, api_key=pubmed_api_key)
    elif source == "semanticscholar":
        all_articles = search_semanticscholar(keywords, start_year, end_year, max_results)
    elif source == "all":
        logger.info(f"Searching all sources. Max results per source: {max_results}")
        # Note: Entrez.email must be set for PubMed search
        if not Entrez.email:
            logger.error("Entrez.email not set. Required for PubMed search if 'all' sources selected.")
            # Decide: skip pubmed, error out, or proceed assuming it might be set by a concurrent call?
            # For now, it will attempt pubmed search which will log its own error if email is missing.

        pubmed_articles = search_pubmed(keywords, start_year, end_year, max_results, api_key=pubmed_api_key)
        semanticscholar_articles = search_semanticscholar(keywords, start_year, end_year, max_results)

        all_articles.extend(pubmed_articles)
        all_articles.extend(semanticscholar_articles)
        # Optional: de-duplicate if necessary, though this can be complex (e.g., based on DOI)
        # For now, just combining.
        logger.info(f"Combined search from 'all' sources yielded {len(all_articles)} articles (before any de-duplication).")
    else:
        logger.warning(f"Unknown literature source: {source}. Returning empty list.")
        return []

    return all_articles


if __name__ == '__main__':
    # Example Usage for testing the module directly
    # Note: Entrez.email must be set for PubMed searches.
    # In a real application, this would be set by the main script.
    # For testing, you can set it here:
    Entrez.email = "your.test.email@example.com" # Replace with a valid email for testing
    # example_pubmed_api_key = "YOUR_PUBMED_API_KEY" # Optional

    test_keywords = "machine learning for drug discovery"
    test_start_year = 2022
    test_end_year = 2023
    test_max_results = 3 # Keep low for testing

    logger.info("\n--- Testing PubMed Search ---")
    # pubmed_results = search_literature("pubmed", test_keywords, test_start_year, test_end_year, test_max_results, pubmed_api_key=None)
    # if pubmed_results:
    #     logger.info(f"Found {len(pubmed_results)} articles from PubMed.")
    #     # for article in pubmed_results:
    #     #     logger.info(f"  Title: {article['Title']} (DOI: {article.get('DOI', 'N/A')})")
    # else:
    #     logger.info("No articles found from PubMed or an error occurred.")

    logger.info("\n--- Testing Semantic Scholar Search ---")
    # s2_results = search_literature("semanticscholar", test_keywords, test_start_year, test_end_year, test_max_results)
    # if s2_results:
    #     logger.info(f"Found {len(s2_results)} articles from Semantic Scholar.")
    #     # for article in s2_results:
    #     #     logger.info(f"  Title: {article['Title']} (DOI: {article.get('DOI', 'N/A')})")
    # else:
    #     logger.info("No articles found from Semantic Scholar or an error occurred.")

    logger.info("\n--- Testing Combined ('all') Search ---")
    # combined_results = search_literature("all", test_keywords, test_start_year, test_end_year, test_max_results, pubmed_api_key=None)
    # if combined_results:
    #     logger.info(f"Found {len(combined_results)} articles from all sources.")
    #     # for article in combined_results:
    #     #     logger.info(f"  Source: {article.get('Source', 'Unknown')}, Title: {article['Title']} (DOI: {article.get('DOI', 'N/A')})")
    # else:
    #     logger.info("No articles found from combined search or an error occurred.")

    # To avoid running actual searches during automated checks if not desired,
    # the actual calls above are commented out.
    # Replace with print statements or keep commented for CI/CD environments.
    logger.info("Example searches in __main__ are set up but commented out to prevent actual API calls during tests.")
    logger.info("Uncomment the search_literature calls above to test live functionality.")
    print("Finished example usage section of literature_search.py.")
