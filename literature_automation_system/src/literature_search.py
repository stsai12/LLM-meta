"""
This module provides functions for searching and retrieving literature from PubMed.
"""
from typing import List, Dict, Optional
from Bio import Entrez, Medline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
    # Entrez.email is expected to be set by the caller (e.g., main.py)
    # before this function is called.
    # For standalone testing of this module, set Entrez.email in the
    # if __name__ == '__main__': block.

    if api_key:
        Entrez.api_key = api_key

    search_term = f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])"

    articles = []

    try:
        logging.info(f"Searching PubMed with query: {search_term}, max_results={max_results}")
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]

        if not pmids:
            logging.info("No articles found for the given query.")
            return []

        logging.info(f"Found {len(pmids)} PMIDs. Fetching details...")
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
        medline_records = Medline.parse(handle)

        for medline_record in medline_records:
            try:
                title = medline_record.get("TI", None)
                authors = medline_record.get("AU", [])
                journal = medline_record.get("JT", None)
                doi = None
                # DOI can be in LID or AID fields
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
                pub_year = pub_year_full[:4] if pub_year_full else None # Extract first 4 chars for year

                abstract = medline_record.get("AB", None)

                articles.append({
                    "Title": title,
                    "Authors": authors,
                    "Journal": journal,
                    "DOI": doi,
                    "Publication Year": pub_year,
                    "Abstract": abstract
                })
            except Exception as e:
                logging.error(f"Error parsing Medline record: {medline_record.get('PMID', 'Unknown PMID')}. Error: {e}")
                continue # Skip this record

        handle.close()
        logging.info(f"Successfully fetched and parsed {len(articles)} articles.")

    except Exception as e:
        logging.error(f"An error occurred during PubMed search or fetch: {e}")
        return [] # Return empty list or partial results if preferred

    return articles

if __name__ == '__main__':
    # Example usage (requires internet connection)
    # Ensure you have Biopython installed: pip install biopython
    # To use NCBI's API more effectively, obtain a free API key:
    # https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Getting_an_API_Key
    # example_api_key = "YOUR_API_KEY" # Replace with your actual API key if you have one

    test_keywords = "artificial intelligence in healthcare"
    test_start_year = 2022
    test_end_year = 2023
    test_max_results = 5 # Small number for quick testing

    print(f"Running example search for '{test_keywords}' from {test_start_year}-{test_end_year}, max {test_max_results} results.")

    # If you have an API key, pass it here:
    # results = search_pubmed(test_keywords, test_start_year, test_end_year, test_max_results, api_key=example_api_key)

    # Example of direct call; ensure Entrez.email is set if you run this part
    # For the project, main.py will handle setting Entrez.email
    # If running this standalone for testing, uncomment and set Entrez.email above.
    # results = search_pubmed(test_keywords, test_start_year, test_end_year, test_max_results)
    results = [] # Avoid running during non-main execution for now

    # if results:
    #     print(f"\nFound {len(results)} articles:")
    # else:
    #     print("Example search in literature_search.py did not run or found no articles.")

    if results: # Keep this structure if you re-enable the direct call above for testing
        print(f"\nFound {len(results)} articles:")
        for i, article in enumerate(results):
            print(f"\n--- Article {i+1} ---")
            print(f"  Title: {article['Title']}")
            print(f"  Authors: {', '.join(article['Authors'])}")
            print(f"  Journal: {article['Journal']}")
            print(f"  DOI: {article['DOI']}")
            print(f"  Publication Year: {article['Publication Year']}")
            print(f"  Abstract: {article['Abstract'][:100]}..." if article['Abstract'] else "N/A")
    else:
        print("No articles found or an error occurred.")
