import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import sys
from typing import List, Dict, Any, Tuple # Added typing
from pathlib import Path # For tmp_path_factory type hint
from pytest import TempPathFactory # For tmp_path_factory type hint
from pytest_mock import MockerFixture # For mocker type hint


# Adjust Python path to import modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_structuring import structure_and_save_data, REVIEW_KEYWORDS


# Sample article data
SAMPLE_ARTICLES_DATA: List[Dict[str, Any]] = [
    {"Title": "A Great Original Study", "Authors": ["Doe, J"], "Journal": "Science", "DOI": "10.1126/science.123", "Publication Year": "2023", "Abstract": "Original abstract."},
    {"Title": "Comprehensive Review of Recent Advances", "Authors": ["Smith, A"], "Journal": "Reviews Quarterly", "DOI": "10.1000/rq.1", "Publication Year": "2022", "Abstract": "This is a review."},
    {"Title": "Another Research Paper", "Authors": ["Lee, B"], "Journal": "Nature", "DOI": "10.1038/nature.xyz", "Publication Year": "2023", "Abstract": "Not a review."},
    {"Title": "A Meta-Analysis of Clinical Trials", "Authors": ["Kim, C"], "Journal": "JAMA", "DOI": "10.1001/jama.123", "Publication Year": "2021", "Abstract": "Meta-analysis results."},
]

@pytest.fixture
def mock_df_to_csv(mocker: MockerFixture) -> MagicMock:
    """Mocks DataFrame.to_csv method."""
    return mocker.patch('pandas.DataFrame.to_csv')

@pytest.fixture
def setup_mocks_for_path(mocker: MockerFixture) -> Tuple[MagicMock, MagicMock, str]:
    """Mocks os.path functions to control output path determination."""
    mock_src_path = '/mock_project_root/src/data_structuring.py'
    mocker.patch('data_structuring.os.path.abspath', return_value=mock_src_path)

    mock_exists = mocker.patch('data_structuring.os.path.exists')
    mock_makedirs = mocker.patch('data_structuring.os.makedirs')
    return mock_exists, mock_makedirs, '/mock_project_root/data'


def test_structure_no_filtering(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    mock_exists, _, expected_data_path_base = setup_mocks_for_path
    mock_exists.return_value = True

    output_filename: str = "test_no_filter.csv"
    result_path: str = structure_and_save_data(SAMPLE_ARTICLES_DATA, output_filename, filter_reviews=False)

    assert result_path == os.path.join(expected_data_path_base, output_filename)
    mock_df_to_csv.assert_called_once()

    df_arg: pd.DataFrame = mock_df_to_csv.call_args[0][0]
    assert len(df_arg) == len(SAMPLE_ARTICLES_DATA)
    assert df_arg["Title"].iloc[0] == "A Great Original Study"

def test_structure_with_filtering(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    mock_exists, _, _ = setup_mocks_for_path
    mock_exists.return_value = True

    output_filename: str = "test_with_filter.csv"
    structure_and_save_data(SAMPLE_ARTICLES_DATA, output_filename, filter_reviews=True)

    mock_df_to_csv.assert_called_once()
    df_arg: pd.DataFrame = mock_df_to_csv.call_args[0][0]

    assert len(df_arg) == 2
    titles: List[str] = df_arg["Title"].tolist()
    assert "A Great Original Study" in titles
    assert "Another Research Paper" in titles
    assert "Comprehensive Review of Recent Advances" not in titles
    assert "A Meta-Analysis of Clinical Trials" not in titles

def test_empty_data_input(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    result_path: str = structure_and_save_data([], "empty.csv", filter_reviews=False)
    assert result_path is None
    mock_df_to_csv.assert_not_called()

def test_all_data_filtered_out(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    reviews_only: List[Dict[str, Any]] = [
        d for d in SAMPLE_ARTICLES_DATA if any(kw.lower() in d["Title"].lower() for kw in REVIEW_KEYWORDS)
    ]
    result_path: str = structure_and_save_data(reviews_only, "all_filtered.csv", filter_reviews=True)
    assert result_path is None
    mock_df_to_csv.assert_not_called()

def test_directory_creation_if_not_exists(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    mock_exists, mock_makedirs, expected_data_path_base = setup_mocks_for_path
    mock_exists.return_value = False

    output_filename: str = "test_dir_creation.csv"
    article_subset: List[Dict[str, Any]] = [SAMPLE_ARTICLES_DATA[0]]

    result_path: str = structure_and_save_data(article_subset, output_filename, filter_reviews=False)

    mock_makedirs.assert_called_once_with(expected_data_path_base)
    assert result_path == os.path.join(expected_data_path_base, output_filename)
    mock_df_to_csv.assert_called_once()

def test_columns_order_and_presence(mock_df_to_csv: MagicMock, setup_mocks_for_path: Tuple[MagicMock, MagicMock, str]) -> None:
    mock_exists, _, _ = setup_mocks_for_path
    mock_exists.return_value = True

    data_with_varying_fields: List[Dict[str, Any]] = [
        {"Title": "T1", "Authors": ["A1"], "Journal": "J1", "DOI": "D1", "Publication Year": "PY1", "Abstract": "Abs1", "ExtraField": "EF1"},
        {"Title": "T2", "Authors": ["A2"], "Journal": "J2", "DOI": "D2", "Publication Year": "PY2"},
    ]
    expected_columns: List[str] = ["Title", "Authors", "Journal", "DOI", "Publication Year", "Abstract"]

    structure_and_save_data(data_with_varying_fields, "varied_fields.csv", filter_reviews=False)

    mock_df_to_csv.assert_called_once()
    df_arg: pd.DataFrame = mock_df_to_csv.call_args[0][0]

    assert list(df_arg.columns) == expected_columns
    assert pd.isna(df_arg[df_arg["Title"] == "T2"]["Abstract"].iloc[0])
    assert "ExtraField" not in df_arg.columns
```
