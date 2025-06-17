import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_structuring import structure_and_save_data

# Sample article data similar to what search_pubmed would return
SAMPLE_ARTICLES_DATA = [
    {
        "Title": "A Study on Topic X", "Authors": ["Author A"], "Journal": "Journal X",
        "DOI": "10.1000/x", "Publication Year": "2023", "Abstract": "Abstract for X."
    },
    {
        "Title": "Review of Topic Y", "Authors": ["Author B"], "Journal": "Journal Y",
        "DOI": "10.1000/y", "Publication Year": "2022", "Abstract": "Review abstract for Y."
    },
    {
        "Title": "Another Study on Topic Z (Meta-analysis)", "Authors": ["Author C"], "Journal": "Journal Z",
        "DOI": "10.1000/z", "Publication Year": "2023", "Abstract": "Meta-analysis abstract for Z."
    },
    {
        "Title": "Systematic Review: Topic A", "Authors": ["Author D"], "Journal": "Journal A",
        "DOI": "10.1000/a", "Publication Year": "2021", "Abstract": "Systematic review of A."
    },
    {
        "Title": "Original Research on B", "Authors": ["Author E"], "Journal": "Journal B",
        "DOI": "10.1000/b", "Publication Year": "2022", "Abstract": "This is not a review."
    }
]

@pytest.fixture
def mock_pandas_to_csv(mocker):
    """Mocks pandas DataFrame.to_csv method."""
    return mocker.patch('pandas.DataFrame.to_csv')

@pytest.fixture
def temp_data_dir(tmp_path_factory):
    """Creates a temporary 'data' directory for testing file output."""
    # Construct path similar to how data_structuring.py does it
    # Assuming tests are run from project root, or this path is adjusted
    # For this test, we'll mock the path construction within structure_and_save_data
    # to ensure it uses our tmp_path.
    return tmp_path_factory.mktemp("data")


def test_structure_and_save_data_no_filtering(mock_pandas_to_csv, temp_data_dir):
    """
    Tests data structuring and saving without review filtering.
    Focuses on DataFrame creation and call to to_csv.
    """
    output_filename = "test_output_no_filter.csv"

    # Mock os.path.join and os.makedirs to control output path for to_csv
    # The goal is to ensure to_csv is called with the correct DataFrame content,
    # and a path that includes our temp_data_dir.

    # We need to mock `os.path.abspath` and `os.path.dirname` for `project_root` calculation
    # inside `structure_and_save_data` if we want to precisely control `data_dir`.
    # Alternatively, and more simply, we can check the arguments passed to `to_csv`.

    with patch('data_structuring.os.path.exists', return_value=True), \
         patch('data_structuring.os.makedirs') as mock_makedirs:

        saved_path = structure_and_save_data(SAMPLE_ARTICLES_DATA, output_filename, filter_reviews=False)

        mock_pandas_to_csv.assert_called_once()
        call_args = mock_pandas_to_csv.call_args[0]
        df_saved = call_args[0] # The DataFrame passed to to_csv

        assert isinstance(df_saved, pd.DataFrame)
        assert len(df_saved) == len(SAMPLE_ARTICLES_DATA)
        assert list(df_saved.columns) == ["Title", "Authors", "Journal", "DOI", "Publication Year", "Abstract"]
        assert df_saved.iloc[0]["Title"] == SAMPLE_ARTICLES_DATA[0]["Title"]

        # Check that the path passed to to_csv is as expected.
        # The function constructs project_root/data/output_filename
        # We can't easily assert the full temp_data_dir path here without more complex mocking of os.path
        # So, we check that the filename part is correct.
        saved_filepath_arg = call_args[1]
        assert output_filename in saved_filepath_arg
        assert "data" in saved_filepath_arg # Ensure it's trying to save in a 'data' directory.

        mock_makedirs.assert_not_called() # Since os.path.exists is True

def test_structure_and_save_data_with_filtering(mock_pandas_to_csv, temp_data_dir):
    """
    Tests data structuring and saving with review filtering enabled.
    """
    output_filename = "test_output_filter.csv"
    # Expected non-review articles: "A Study on Topic X", "Original Research on B"
    expected_non_review_count = 2

    with patch('data_structuring.os.path.exists', return_value=True):
        saved_path = structure_and_save_data(SAMPLE_ARTICLES_DATA, output_filename, filter_reviews=True)

        mock_pandas_to_csv.assert_called_once()
        df_saved = mock_pandas_to_csv.call_args[0][0]

        assert isinstance(df_saved, pd.DataFrame)
        assert len(df_saved) == expected_non_review_count
        assert "Review of Topic Y" not in df_saved["Title"].tolist()
        assert "Another Study on Topic Z (Meta-analysis)" not in df_saved["Title"].tolist()
        assert "Systematic Review: Topic A" not in df_saved["Title"].tolist()
        assert "A Study on Topic X" in df_saved["Title"].tolist()
        assert "Original Research on B" in df_saved["Title"].tolist()

def test_structure_and_save_data_empty_input(mock_pandas_to_csv):
    """
    Tests behavior with empty input article data.
    """
    saved_path = structure_and_save_data([], "empty.csv")
    assert saved_path is None
    mock_pandas_to_csv.assert_not_called()

def test_structure_and_save_data_all_filtered_out(mock_pandas_to_csv):
    """
    Tests behavior when all articles are filtered out.
    """
    review_only_articles = [
        article for article in SAMPLE_ARTICLES_DATA if "review" in article["Title"].lower() or "meta-analysis" in article["Title"].lower()
    ]
    saved_path = structure_and_save_data(review_only_articles, "all_filtered.csv", filter_reviews=True)
    assert saved_path is None
    mock_pandas_to_csv.assert_not_called()


def test_directory_creation(mocker, temp_data_dir):
    """
    Test that the data directory is created if it doesn't exist.
    This test is a bit more involved due to how paths are constructed.
    """
    output_filename = "test_dir_creation.csv"

    # Get the path of the 'src' directory to correctly mock project_root calculation
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
    project_root_mock = os.path.dirname(src_dir) # This is <repo_root>/literature_automation_system

    # The 'data' directory that structure_and_save_data will try to create
    expected_data_dir = os.path.join(project_root_mock, "data")

    # Mock os.path.abspath to control the base for __file__
    # This helps in predicting `project_root` inside `structure_and_save_data`
    mocker.patch('data_structuring.os.path.abspath', lambda x: os.path.join(src_dir, 'data_structuring.py'))

    mock_exists = mocker.patch('data_structuring.os.path.exists')
    mock_makedirs = mocker.patch('data_structuring.os.makedirs')
    mock_df_to_csv = mocker.patch('pandas.DataFrame.to_csv')

    # First, simulate directory does not exist
    mock_exists.return_value = False

    structure_and_save_data([SAMPLE_ARTICLES_DATA[0]], output_filename, filter_reviews=False)

    # Check that os.makedirs was called with the expected data directory path
    mock_makedirs.assert_called_once_with(expected_data_dir)

    # Check that to_csv was called with a path inside the expected data_dir
    mock_df_to_csv.assert_called_once()
    saved_filepath_arg = mock_df_to_csv.call_args[0][1]
    assert saved_filepath_arg == os.path.join(expected_data_dir, output_filename)

    # Reset mocks for next scenario
    mock_exists.reset_mock()
    mock_makedirs.reset_mock()
    mock_df_to_csv.reset_mock()

    # Second, simulate directory already exists
    mock_exists.return_value = True
    structure_and_save_data([SAMPLE_ARTICLES_DATA[0]], output_filename, filter_reviews=False)
    mock_makedirs.assert_not_called() # Should not be called if dir exists
    mock_df_to_csv.assert_called_once()


# To actually test CSV content, you would not mock to_csv but let it write to temp_data_dir
# and then read the file. Example:
# def test_csv_content_actual_write(temp_data_dir):
#     articles = [SAMPLE_ARTICLES_DATA[0]]
#     output_filename = "actual_content.csv"
#
#     # Need to make data_structuring.py use temp_data_dir as its 'data' parent
#     # This requires careful mocking of os.path.dirname and os.path.abspath
#     # or modifying structure_and_save_data to accept a base_path for 'data'
#
#     # For simplicity, if structure_and_save_data could take a base_dir:
#     # saved_path = structure_and_save_data(articles, output_filename, filter_reviews=False, base_data_dir=temp_data_dir)
#     # assert saved_path == os.path.join(temp_data_dir, output_filename)
#     # assert os.path.exists(saved_path)
#     # df_read = pd.read_csv(saved_path)
#     # assert len(df_read) == 1
#     # assert df_read.iloc[0]["Title"] == articles[0]["Title"]
#
#     # Given the current implementation, this is harder to test cleanly without refactoring structure_and_save_data
#     pass

```python
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust Python path to import modules from src
# This assumes tests are in 'tests/' and src is a sibling directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_structuring import structure_and_save_data, REVIEW_KEYWORDS

# Sample article data
SAMPLE_ARTICLES_DATA_FOR_STRUCTURING = [
    {"Title": "A Great Original Study", "Authors": ["Doe, J"], "Journal": "Science", "DOI": "10.1126/science.123", "Publication Year": "2023", "Abstract": "Original abstract."},
    {"Title": "Comprehensive Review of Recent Advances", "Authors": ["Smith, A"], "Journal": "Reviews Quarterly", "DOI": "10.1000/rq.1", "Publication Year": "2022", "Abstract": "This is a review."},
    {"Title": "Another Research Paper", "Authors": ["Lee, B"], "Journal": "Nature", "DOI": "10.1038/nature.xyz", "Publication Year": "2023", "Abstract": "Not a review."},
    {"Title": "A Meta-Analysis of Clinical Trials", "Authors": ["Kim, C"], "Journal": "JAMA", "DOI": "10.1001/jama.123", "Publication Year": "2021", "Abstract": "Meta-analysis results."},
]

@pytest.fixture
def mock_df_to_csv(mocker):
    """Mocks DataFrame.to_csv method."""
    return mocker.patch('pandas.DataFrame.to_csv')

@pytest.fixture
def setup_mocks_for_path(mocker):
    """Mocks os.path functions to control output path determination."""
    # Mock abspath to control where 'data_structuring.py' thinks it is.
    # This influences how `project_root` and `data_dir` are calculated.
    # Let's assume 'data_structuring.py' is in '/mock_project_root/src/data_structuring.py'
    mock_src_path = '/mock_project_root/src/data_structuring.py'
    mocker.patch('data_structuring.os.path.abspath', return_value=mock_src_path)

    # Mock os.path.exists and os.makedirs
    mock_exists = mocker.patch('data_structuring.os.path.exists')
    mock_makedirs = mocker.patch('data_structuring.os.makedirs')
    return mock_exists, mock_makedirs, '/mock_project_root/data'


def test_structure_no_filtering(mock_df_to_csv, setup_mocks_for_path):
    mock_exists, _, expected_data_path_base = setup_mocks_for_path
    mock_exists.return_value = True # Assume data directory exists

    output_filename = "test_no_filter.csv"
    result_path = structure_and_save_data(SAMPLE_ARTICLES_DATA_FOR_STRUCTURING, output_filename, filter_reviews=False)

    assert result_path == os.path.join(expected_data_path_base, output_filename)
    mock_df_to_csv.assert_called_once()

    df_arg = mock_df_to_csv.call_args[0][0]
    assert len(df_arg) == len(SAMPLE_ARTICLES_DATA_FOR_STRUCTURING)
    assert df_arg["Title"].iloc[0] == "A Great Original Study"

def test_structure_with_filtering(mock_df_to_csv, setup_mocks_for_path):
    mock_exists, _, _ = setup_mocks_for_path
    mock_exists.return_value = True

    output_filename = "test_with_filter.csv"
    structure_and_save_data(SAMPLE_ARTICLES_DATA_FOR_STRUCTURING, output_filename, filter_reviews=True)

    mock_df_to_csv.assert_called_once()
    df_arg = mock_df_to_csv.call_args[0][0]

    # Expected: "A Great Original Study", "Another Research Paper"
    assert len(df_arg) == 2
    titles = df_arg["Title"].tolist()
    assert "A Great Original Study" in titles
    assert "Another Research Paper" in titles
    assert "Comprehensive Review of Recent Advances" not in titles
    assert "A Meta-Analysis of Clinical Trials" not in titles

def test_empty_data_input(mock_df_to_csv, setup_mocks_for_path):
    result_path = structure_and_save_data([], "empty.csv", filter_reviews=False)
    assert result_path is None
    mock_df_to_csv.assert_not_called()

def test_all_data_filtered_out(mock_df_to_csv, setup_mocks_for_path):
    reviews_only = [
        d for d in SAMPLE_ARTICLES_DATA_FOR_STRUCTURING if any(kw.lower() in d["Title"].lower() for kw in REVIEW_KEYWORDS)
    ]
    result_path = structure_and_save_data(reviews_only, "all_filtered.csv", filter_reviews=True)
    assert result_path is None
    mock_df_to_csv.assert_not_called()

def test_directory_creation_if_not_exists(mock_df_to_csv, setup_mocks_for_path):
    mock_exists, mock_makedirs, expected_data_path_base = setup_mocks_for_path
    mock_exists.return_value = False # Simulate data directory does not exist

    output_filename = "test_dir_creation.csv"
    article_subset = [SAMPLE_ARTICLES_DATA_FOR_STRUCTURING[0]] # Just one article

    result_path = structure_and_save_data(article_subset, output_filename, filter_reviews=False)

    mock_makedirs.assert_called_once_with(expected_data_path_base)
    assert result_path == os.path.join(expected_data_path_base, output_filename)
    mock_df_to_csv.assert_called_once()

def test_columns_order_and_presence(mock_df_to_csv, setup_mocks_for_path):
    mock_exists, _, _ = setup_mocks_for_path
    mock_exists.return_value = True

    # Sample data with potentially missing or extra fields
    data_with_varying_fields = [
        {"Title": "T1", "Authors": ["A1"], "Journal": "J1", "DOI": "D1", "Publication Year": "PY1", "Abstract": "Abs1", "ExtraField": "EF1"},
        {"Title": "T2", "Authors": ["A2"], "Journal": "J2", "DOI": "D2", "Publication Year": "PY2"}, # No Abstract
    ]
    expected_columns = ["Title", "Authors", "Journal", "DOI", "Publication Year", "Abstract"]

    structure_and_save_data(data_with_varying_fields, "varied_fields.csv", filter_reviews=False)

    mock_df_to_csv.assert_called_once()
    df_arg = mock_df_to_csv.call_args[0][0]

    assert list(df_arg.columns) == expected_columns
    # Check that 'Abstract' column exists and has None/NaN for the second entry
    assert pd.isna(df_arg[df_arg["Title"] == "T2"]["Abstract"].iloc[0])
    assert "ExtraField" not in df_arg.columns
```
