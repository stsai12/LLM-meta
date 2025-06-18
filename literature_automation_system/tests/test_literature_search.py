import pytest
from unittest.mock import patch, MagicMock
from Bio import Entrez, Medline
import sys
import os
from typing import Dict, List, Any, Generator, Optional # Added typing
from pytest_mock import MockerFixture # For mocker type hint

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
PARSED_SAMPLE_RECORD_1: Dict[str, Any] = {
    "PMID": "12345678",
    "TI": "Test Title for Article 1",
    "AU": ["Author A", "Author B"],
    "JT": "Journal of Testing",
    "DP": "2023 Jan",
    "AB": "This is the abstract for article 1. It mentions some keywords.",
    "LID": "10.1234/jtest.123 [doi]"
}
PARSED_SAMPLE_RECORD_2: Dict[str, Any] = {
    "PMID": "87654321",
    "TI": "Test Title for Article 2 - No DOI",
    "AU": ["Author C"],
    "JT": "Another Journal",
    "DP": "2022"
}


@pytest.fixture
def mock_entrez_email() -> Generator[None, None, None]:
    # Set Entrez.email for testing purposes, as it's normally set by main.py
    original_email: Optional[str] = Entrez.email
    Entrez.email = "test.user@example.com"
    yield
    Entrez.email = original_email

def test_search_pubmed_parsing(mocker: MockerFixture, mock_entrez_email: None) -> None:
    """
    Tests if search_pubmed correctly parses mock Medline data.
    """
    # Mock Entrez.esearch
    mock_esearch_handle = MagicMock()
    # Entrez.esearch returns a handle, which .read() gives XML string.
    # Entrez.read then parses this XML string.
    # So, first mock the handle's read method if we were testing Entrez.read directly.
    # However, we are mocking Entrez.read itself.

    # Mock Entrez.read to return the parsed dictionary directly
    mocker.patch('Bio.Entrez.read', return_value={"IdList": ["12345678", "87654321"]})

    # Mock Entrez.esearch to return a dummy handle (its read method won't be used due to mocking Entrez.read)
    mocker.patch('Bio.Entrez.esearch', return_value=MagicMock())

    # Mock Medline.parse to return our pre-parsed records
    mock_medline_parse = mocker.patch('Bio.Medline.parse', return_value=[PARSED_SAMPLE_RECORD_1, PARSED_SAMPLE_RECORD_2])

    # Mock Entrez.efetch to return a dummy handle (Medline.parse consumes this handle)
    mocker.patch('Bio.Entrez.efetch', return_value=MagicMock())

    keywords: str = "test keywords"
    start_year: int = 2022
    end_year: int = 2023
    max_results: int = 2

    results: List[Dict[str, Any]] = search_pubmed(keywords, start_year, end_year, max_results)

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
    assert results[1]["DOI"] is None
    assert results[1]["Publication Year"] == "2022"
    assert results[1]["Abstract"] is None

    mock_medline_parse.assert_called_once()

    Entrez.esearch.assert_called_once_with(
        db="pubmed",
        term=f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])",
        retmax=str(max_results),
        sort="relevance"
    )


def test_search_pubmed_query_construction(mocker: MockerFixture, mock_entrez_email: None) -> None:
    """
    Tests how search_pubmed constructs the query string.
    """
    mock_esearch = mocker.patch('Bio.Entrez.esearch')
    mocker.patch('Bio.Entrez.read', return_value={'IdList': []})
    mocker.patch('Bio.Entrez.efetch')
    mocker.patch('Bio.Medline.parse', return_value=[])


    keywords: str = "specific query terms"
    start_year: int = 2020
    end_year: int = 2021
    max_r: int = 5

    search_pubmed(keywords, start_year, end_year, max_r)

    mock_esearch.assert_called_once_with(
        db="pubmed",
        term=f"{keywords} AND ({start_year}[mindate] : {end_year}[maxdate])",
        retmax=str(max_r),
        sort="relevance"
    )

def test_search_pubmed_no_results(mocker: MockerFixture, mock_entrez_email: None) -> None:
    """
    Tests behavior when Entrez.esearch returns no PMIDs.
    """
    mocker.patch('Bio.Entrez.read', return_value={"IdList": []})
    mocker.patch('Bio.Entrez.esearch', return_value=MagicMock())
    mock_efetch = mocker.patch('Bio.Entrez.efetch')

    results: List[Dict[str, Any]] = search_pubmed("empty search", 2022, 2023, 10)

    assert results == []
    mock_efetch.assert_not_called()

def test_search_pubmed_api_key(mocker: MockerFixture, mock_entrez_email: None) -> None:
    """
    Tests that the API key is correctly assigned to Entrez.api_key.
    """
    original_api_key: Optional[str] = Entrez.api_key if hasattr(Entrez, 'api_key') else None

    mocker.patch('Bio.Entrez.esearch')
    mocker.patch('Bio.Entrez.read', return_value={'IdList': []})
    mocker.patch('Bio.Entrez.efetch')
    mocker.patch('Bio.Medline.parse', return_value=[])

    test_api_key: str = "test_api_key_12345"
    search_pubmed("keywords", 2020, 2021, 1, api_key=test_api_key)

    assert Entrez.api_key == test_api_key

    Entrez.api_key = original_api_key
    if original_api_key is None and hasattr(Entrez, 'api_key'):
        delattr(Entrez, 'api_key')

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
