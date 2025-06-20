import pytest
from unittest.mock import patch, MagicMock
from Bio import Entrez, Medline
import sys
import os
from typing import Dict, List, Any, Generator, Optional # Added typing
from pytest_mock import MockerFixture # For mocker type hint

# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import requests # Ensure requests is imported
from literature_search import search_pubmed, search_semanticscholar, search_literature

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

# --- Helper Functions and Fixtures for Semantic Scholar Tests ---

def sample_s2_api_response(num_articles: int = 1) -> Dict[str, Any]:
    """
    Generates a sample Semantic Scholar API response dictionary.
    """
    articles_data = []
    if num_articles > 0:
        for i in range(num_articles):
            articles_data.append({
                "paperId": f"s2id{i+1}",
                "title": f"Sample S2 Title {i+1}",
                "authors": [{"name": f"S2 Author A{i+1}"}, {"name": f"S2 Author B{i+1}"}],
                "year": 2023 + i,
                "abstract": f"This is a sample abstract for S2 article {i+1}.",
                "journal": {"name": f"S2 Journal {i+1}"},
                "externalIds": {"DOI": f"10.s2example/s2.doi.{i+1}"}
            })
    return {"data": articles_data, "total": num_articles, "offset": 0, "next": 0}


@pytest.fixture
def mock_requests_get(mocker: MockerFixture) -> MagicMock:
    """Mocks requests.get for Semantic Scholar calls."""
    return mocker.patch('requests.get')

# --- New Test Functions ---

def test_search_semanticscholar_success(mock_requests_get: MagicMock) -> None:
    """Tests a successful call to search_semanticscholar."""
    mock_response = MagicMock()
    mock_response.json.return_value = sample_s2_api_response(1)
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    results = search_semanticscholar(keywords="test query", start_year=2022, end_year=2023, max_results=1)

    assert len(results) == 1
    article = results[0]
    assert article["Title"] == "Sample S2 Title 1"
    assert article["Authors"] == ["S2 Author A1", "S2 Author B1"]
    assert article["Journal"] == "S2 Journal 1"
    assert article["DOI"] == "10.s2example/s2.doi.1"
    assert article["Publication Year"] == "2023"
    assert article["Abstract"] == "This is a sample abstract for S2 article 1."
    assert article["Source"] == "Semantic Scholar"

    mock_requests_get.assert_called_once()
    call_args = mock_requests_get.call_args
    assert call_args.kwargs['params']['query'] == "test query"
    assert call_args.kwargs['params']['limit'] == 1
    assert call_args.kwargs['params']['year'] == "2022-2023"
    # Check that API key is NOT in headers if not provided
    if 'headers' in call_args.kwargs and call_args.kwargs['headers']:
        assert 'x-api-key' not in call_args.kwargs['headers']


def test_search_semanticscholar_with_api_key(mock_requests_get: MagicMock) -> None:
    """Tests search_semanticscholar with an API key."""
    mock_response = MagicMock()
    mock_response.json.return_value = sample_s2_api_response(1)
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    test_api_key = "test_s2_key_xyz"
    search_semanticscholar(keywords="test query", start_year=2022, end_year=2023, max_results=1, api_key=test_api_key)

    mock_requests_get.assert_called_once()
    call_args = mock_requests_get.call_args
    assert 'headers' in call_args.kwargs
    assert call_args.kwargs['headers']['x-api-key'] == test_api_key


def test_search_semanticscholar_http_error(mock_requests_get: MagicMock, caplog: Any) -> None:
    """Tests search_semanticscholar handling of HTTP errors."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error: Too Many Requests for url")
    mock_response.text = '{"message": "Too Many Requests."}' # Simulate error response body
    mock_requests_get.return_value = mock_response

    results = search_semanticscholar(keywords="query", start_year=2022, end_year=2023, max_results=5)

    assert len(results) == 0
    assert "HTTP error occurred while searching Semantic Scholar: 429 Client Error" in caplog.text
    assert '{"message": "Too Many Requests."}' in caplog.text


def test_search_semanticscholar_no_results(mock_requests_get: MagicMock) -> None:
    """Tests search_semanticscholar when API returns no articles."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [], "total": 0} # Empty data list
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    results = search_semanticscholar(keywords="rare query", start_year=2022, end_year=2023, max_results=5)

    assert len(results) == 0


@patch('literature_search.search_semanticscholar')
@patch('literature_search.search_pubmed')
def test_search_literature_source_pubmed(
    mock_pubmed_search: MagicMock,
    mock_s2_search: MagicMock,
    mock_entrez_email: None # Fixture to set Entrez.email
) -> None:
    """Tests search_literature dispatcher for 'pubmed' source."""
    mock_pubmed_search.return_value = [{"Title": "PubMed Article", "Source": "PubMed"}]

    results = search_literature(
        source="pubmed", keywords="query", start_year=2022, end_year=2023, max_results=10,
        pubmed_api_key="pkey123", semanticscholar_api_key="s2key456"
    )

    mock_pubmed_search.assert_called_once_with("query", 2022, 2023, 10, api_key="pkey123")
    mock_s2_search.assert_not_called()
    assert len(results) == 1
    assert results[0]["Source"] == "PubMed"


@patch('literature_search.search_semanticscholar')
@patch('literature_search.search_pubmed')
def test_search_literature_source_semanticscholar(
    mock_pubmed_search: MagicMock,
    mock_s2_search: MagicMock
) -> None:
    """Tests search_literature dispatcher for 'semanticscholar' source."""
    mock_s2_search.return_value = [{"Title": "S2 Article", "Source": "Semantic Scholar"}]

    results = search_literature(
        source="semanticscholar", keywords="query", start_year=2022, end_year=2023, max_results=10,
        pubmed_api_key="pkey123", semanticscholar_api_key="s2key456"
    )

    mock_s2_search.assert_called_once_with("query", 2022, 2023, 10, api_key="s2key456")
    mock_pubmed_search.assert_not_called()
    assert len(results) == 1
    assert results[0]["Source"] == "Semantic Scholar"


@patch('literature_search.search_semanticscholar')
@patch('literature_search.search_pubmed')
def test_search_literature_source_all(
    mock_pubmed_search: MagicMock,
    mock_s2_search: MagicMock,
    mock_entrez_email: None # Fixture to set Entrez.email
) -> None:
    """Tests search_literature dispatcher for 'all' sources."""
    mock_pubmed_search.return_value = [{"Title": "PubMed Article", "DOI": "doi1", "Source": "PubMed"}]
    mock_s2_search.return_value = [{"Title": "S2 Article", "DOI": "doi2", "Source": "Semantic Scholar"}]

    results = search_literature(
        source="all", keywords="query", start_year=2022, end_year=2023, max_results=10,
        pubmed_api_key="pkey123", semanticscholar_api_key="s2key456"
    )

    mock_pubmed_search.assert_called_once_with("query", 2022, 2023, 10, api_key="pkey123")
    mock_s2_search.assert_called_once_with("query", 2022, 2023, 10, api_key="s2key456")
    assert len(results) == 2
    assert any(r["Source"] == "PubMed" for r in results)
    assert any(r["Source"] == "Semantic Scholar" for r in results)


@patch('literature_search.search_semanticscholar')
@patch('literature_search.search_pubmed')
def test_search_literature_source_unknown(
    mock_pubmed_search: MagicMock,
    mock_s2_search: MagicMock,
    caplog: Any
) -> None:
    """Tests search_literature dispatcher for an unknown source."""
    results = search_literature(
        source="unknown_source", keywords="query", start_year=2022, end_year=2023, max_results=10
    )

    mock_pubmed_search.assert_not_called()
    mock_s2_search.assert_not_called()
    assert len(results) == 0
    assert "Unknown literature source: unknown_source" in caplog.text


def test_search_pubmed_new_date_query_construction(mocker: MockerFixture, mock_entrez_email: None) -> None:
    """
    Tests that search_pubmed constructs the query string with the new explicit date format.
    """
    mock_esearch = mocker.patch('Bio.Entrez.esearch')
    # Mock further calls to prevent actual API interaction or processing
    mocker.patch('Bio.Entrez.read', return_value={'IdList': []})
    mocker.patch('Bio.Entrez.efetch')
    mocker.patch('Bio.Medline.parse', return_value=[])

    keywords = "updated query syntax"
    start_year = 2021
    end_year = 2022
    max_r = 7

    search_pubmed(keywords, start_year, end_year, max_r)

    expected_term = f"{keywords} AND ('{start_year}/01/01'[Date - Publication] : '{end_year}/12/31'[Date - Publication])"
    mock_esearch.assert_called_once_with(
        db="pubmed",
        term=expected_term,
        retmax=str(max_r),
        sort="relevance"
    )
