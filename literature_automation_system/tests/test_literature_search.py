import pytest
from unittest.mock import patch, MagicMock # mocker can be used if preferred via pytest-mock
from Bio import Entrez, Medline
import sys
import os

# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from literature_search import search_pubmed

# Sample Medline record string (simplified)
SAMPLE_MEDLINE_RECORD = """
PMID- 12345678
TI  - Test Title for Article 1
AU  - Author A
AU  - Author B
JT  - Journal of Testing
DP  - 2023 Jan
AB  - This is the abstract for article 1. It mentions some keywords.
LID - 10.1234/jtest.123 [doi]
"""

SAMPLE_MEDLINE_RECORD_NO_DOI_OR_ABSTRACT = """
PMID- 87654321
TI  - Test Title for Article 2 - No DOI
AU  - Author C
JT  - Another Journal
DP  - 2022
"""

# This is what Medline.parse would return for the above string
PARSED_SAMPLE_RECORD_1 = {
    "PMID": "12345678",
    "TI": "Test Title for Article 1",
    "AU": ["Author A", "Author B"],
    "JT": "Journal of Testing",
    "DP": "2023 Jan",
    "AB": "This is the abstract for article 1. It mentions some keywords.",
    "LID": "10.1234/jtest.123 [doi]"
}
PARSED_SAMPLE_RECORD_2 = {
    "PMID": "87654321",
    "TI": "Test Title for Article 2 - No DOI",
    "AU": ["Author C"],
    "JT": "Another Journal",
    "DP": "2022"
}


@pytest.fixture
def mock_entrez_email():
    # Set Entrez.email for testing purposes, as it's normally set by main.py
    original_email = Entrez.email
    Entrez.email = "test.user@example.com"
    yield
    Entrez.email = original_email

def test_search_pubmed_parsing(mocker, mock_entrez_email):
    """
    Tests if search_pubmed correctly parses mock Medline data.
    """
    # Mock Entrez.esearch
    mock_esearch_handle = MagicMock()
    mock_esearch_handle.read.return_value = '<?xml version="1.0"?><eSearchResult><IdList><Id>12345678</Id><Id>87654321</Id></IdList></eSearchResult>'

    # We need Entrez.read to parse the XML from esearch
    # The actual Entrez.read function will parse the XML string into a dictionary
    mocker.patch('Bio.Entrez.read', return_value={"IdList": ["12345678", "87654321"]})

    # Mock Entrez.efetch to return our sample Medline data
    mock_efetch_handle = MagicMock()
    # Medline.parse expects an iterable (like a file handle line by line)
    # For simplicity, we'll mock Medline.parse directly

    mocker.patch('Bio.Entrez.esearch', return_value=mock_esearch_handle)

    # Mock Medline.parse to return our pre-parsed records
    # This is easier than crafting a perfect Medline text string for efetch's handle
    mock_medline_parse = mocker.patch('Bio.Medline.parse', return_value=[PARSED_SAMPLE_RECORD_1, PARSED_SAMPLE_RECORD_2])

    # Mock Entrez.efetch itself, though its handle's behavior is more critical
    # and we are mocking Medline.parse which consumes the handle
    mocker.patch('Bio.Entrez.efetch', return_value=MagicMock()) # efetch returns a handle

    keywords = "test keywords"
    start_year = 2022
    end_year = 2023
    max_results = 2

    results = search_pubmed(keywords, start_year, end_year, max_results)

    assert len(results) == 2

    # Article 1
    assert results[0]["Title"] == "Test Title for Article 1"
    assert results[0]["Authors"] == ["Author A", "Author B"]
    assert results[0]["Journal"] == "Journal of Testing"
    assert results[0]["DOI"] == "10.1234/jtest.123"
    assert results[0]["Publication Year"] == "2023"
    assert results[0]["Abstract"] == "This is the abstract for article 1. It mentions some keywords."

    # Article 2
    assert results[1]["Title"] == "Test Title for Article 2 - No DOI"
    assert results[1]["Authors"] == ["Author C"]
    assert results[1]["DOI"] is None # Check for None if DOI is missing
    assert results[1]["Publication Year"] == "2022"
    assert results[1]["Abstract"] is None

    # Check that Medline.parse was called (implicitly means efetch handle was used)
    mock_medline_parse.assert_called_once()

    # Check esearch call arguments (Entrez.read is mocked, so we check its input)
    # Entrez.esearch was called with these args
    Entrez.esearch.assert_called_once_with(
        db="pubmed",
        term=f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])",
        retmax=str(max_results),
        sort="relevance"
    )


def test_search_pubmed_query_construction(mocker, mock_entrez_email):
    """
    Tests how search_pubmed constructs the query string.
    """
    mock_esearch = mocker.patch('Bio.Entrez.esearch')
    # Mock Entrez.read to prevent actual parsing and further calls
    mocker.patch('Bio.Entrez.read', return_value={'IdList': []}) # Return empty list to stop processing
    # Mock efetch and Medline.parse as they are not relevant for this specific test
    mocker.patch('Bio.Entrez.efetch')
    mocker.patch('Bio.Medline.parse', return_value=[])


    keywords = "specific query terms"
    start_year = 2020
    end_year = 2021
    max_r = 5

    search_pubmed(keywords, start_year, end_year, max_r)

    mock_esearch.assert_called_once_with(
        db="pubmed",
        term=f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])",
        retmax=str(max_r),
        sort="relevance"
    )

def test_search_pubmed_no_results(mocker, mock_entrez_email):
    """
    Tests behavior when Entrez.esearch returns no PMIDs.
    """
    mocker.patch('Bio.Entrez.read', return_value={"IdList": []}) # No IDs found
    mock_esearch_handle = MagicMock() # Dummy handle
    mocker.patch('Bio.Entrez.esearch', return_value=mock_esearch_handle)
    mock_efetch = mocker.patch('Bio.Entrez.efetch') # Should not be called

    results = search_pubmed("empty search", 2022, 2023, 10)

    assert results == []
    mock_efetch.assert_not_called()

def test_search_pubmed_api_key(mocker, mock_entrez_email):
    """
    Tests that the API key is correctly assigned to Entrez.api_key.
    """
    # Store original api_key if exists, and restore it later
    original_api_key = Entrez.api_key if hasattr(Entrez, 'api_key') else None

    mock_esearch = mocker.patch('Bio.Entrez.esearch')
    mocker.patch('Bio.Entrez.read', return_value={'IdList': []})
    mocker.patch('Bio.Entrez.efetch')
    mocker.patch('Bio.Medline.parse', return_value=[])

    test_api_key = "test_api_key_12345"
    search_pubmed("keywords", 2020, 2021, 1, api_key=test_api_key)

    assert Entrez.api_key == test_api_key

    # Clean up: restore original api_key
    Entrez.api_key = original_api_key
    if original_api_key is None and hasattr(Entrez, 'api_key'):
        delattr(Entrez, 'api_key') # if it was not set before

# Example of how you might load Medline data from a file if needed:
# @pytest.fixture
# def sample_medline_data_from_file():
#     path = os.path.join(os.path.dirname(__file__), "sample_medline_data.txt")
#     with open(path, 'r') as f:
#         return f.read()

# To use the above fixture:
# def test_with_file_data(mocker, sample_medline_data_from_file):
#     mocker.patch('Bio.Entrez.efetch', return_value=io.StringIO(sample_medline_data_from_file))
#     # ... rest of the test
