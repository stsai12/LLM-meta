# Automated Literature Search and Preprocessing System

## Overview
This system automates the process of searching for scientific literature on PubMed, structuring the retrieved metadata, and optionally extracting key information from article abstracts using a Language Model (LLM). It is designed to help researchers and analysts quickly gather and preprocess relevant literature for their studies.

## Features
- Connects to PubMed via the Entrez API using Biopython.
- Fetches comprehensive article metadata: title, authors, journal, DOI, publication year, and abstract.
- Allows user-defined search criteria including keywords, publication year range, and maximum number of results.
- Optionally filters out review articles and meta-analyses based on keywords in titles.
- Saves the collected and processed data into a structured CSV file.
- Optional LLM-based preprocessing (via Hugging Face `transformers`) to extract specific details from abstracts:
    - Sample size
    - Study design
    - Outcome type
    - Intervention or exposure

## Project Structure
```
literature_automation_system/
├── data/                 # Output CSV files are saved here
├── src/                  # Source code modules
│   ├── literature_search.py
│   ├── data_structuring.py
│   ├── llm_preprocessing.py
│   └── __init__.py       # Makes 'src' a Python package
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup
1.  **Download the files or clone the repository.**
    If you have Git installed, you can clone the repository:
    ```bash
    # git clone <repository_url> # Replace <repository_url> if applicable
    # cd literature_automation_system
    ```
    Otherwise, download the `literature_automation_system` directory and its contents.

2.  **Create a virtual environment (recommended):**
    This isolates project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```

3.  **Install dependencies:**
    Ensure your virtual environment is activated.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The LLM module uses Hugging Face's `transformers` library. The first time you run the system with LLM preprocessing enabled (`--enable_llm`), it will download the specified pre-trained model (e.g., "distilbert-base-cased-distilled-squad"). This download can be several hundred MBs and may take some time depending on your internet connection.*

4.  **NCBI Entrez Email:**
    PubMed's Entrez API requires users to specify an email address for API access. This helps NCBI contact users if there are issues. You **must** set this via the `--entrez_email` argument when running `main.py`.
    Example: `--entrez_email your.name@example.com`

5.  **NCBI API Key (Optional but Recommended):**
    For more intensive use or to avoid potential rate-limiting, it is highly recommended to obtain a free NCBI API key. Once you have a key, provide it using the `--ncbi_api_key YOUR_KEY_HERE` argument.
    You can get an API key from your NCBI account settings.

## Usage
The main script `main.py` is used to run the entire pipeline from the command line. Ensure your virtual environment is activated before running.

**Required arguments:**
- `keywords`: The search query string (e.g., "covid vaccine efficacy"). Enclose in quotes if it contains spaces.
- `--start_year`: The starting year for the publication search range (integer).
- `--end_year`: The ending year for the publication search range (integer).

**Optional arguments:**
- `--max_results INT`: Maximum number of articles to fetch (default: 10).
- `--output_file FILENAME.csv`: Name for the output CSV file (default: "literature_results.csv"). This file will be saved in the `data/` directory.
- `--filter_reviews`: A flag; if present, the system will attempt to filter out review articles and meta-analyses.
- `--enable_llm`: A flag; if present, enables LLM-based preprocessing of abstracts to extract additional fields.
- `--llm_model_name MODEL_NAME`: Specify the Hugging Face model name for LLM preprocessing (default: "distilbert-base-cased-distilled-squad").
- `--entrez_email EMAIL_ADDRESS`: **(Effectively Required)** Your email address for NCBI Entrez.
- `--ncbi_api_key YOUR_API_KEY`: Your NCBI API key (optional, but recommended).

**Example commands:**

1.  **Basic search (remember to set your email):**
    ```bash
    python main.py "diabetes management" --start_year 2020 --end_year 2023 --entrez_email user@example.com
    ```
2.  **Search with more results, custom output file name, and API key:**
    ```bash
    python main.py "cancer immunotherapy" --start_year 2021 --end_year 2023 --max_results 50 --output_file cancer_immuno_research.csv --entrez_email user@example.com --ncbi_api_key YOUR_NCBI_KEY
    ```
3.  **Search, filter reviews, and enable LLM preprocessing with a specific model:**
    ```bash
    python main.py "Alzheimer's early detection biomarkers" --start_year 2019 --end_year 2022 --max_results 25 --filter_reviews --enable_llm --llm_model_name "dmis-lab/biobert-base-cased-v1.1-squad" --entrez_email user@example.com --ncbi_api_key YOUR_NCBI_KEY
    ```
    *(Note: Using a different LLM model like BioBERT might require ensuring its compatibility with the question-answering pipeline and could have different performance characteristics.)*

## Modules

### 1. Literature Search (`src/literature_search.py`)
- Utilizes `Biopython` to interact with the NCBI Entrez API.
- Constructs queries to PubMed based on keywords, date range, and desired number of results.
- Fetches article details including Title, Authors (AU), Journal (JT), DOI (LID/AID), Publication Year (DP), and Abstract (AB).

### 2. Data Structuring (`src/data_structuring.py`)
- Receives the list of article dictionaries from the search module.
- If the `--filter_reviews` flag is active, it inspects article titles for keywords like "review", "meta-analysis", etc., and excludes matching articles.
- Employs `pandas` to convert the list of (filtered) articles into a DataFrame.
- Saves this DataFrame to a CSV file in the `data/` directory.

### 3. LLM Preprocessing (`src/llm_preprocessing.py`) (Optional)
- Activated by the `--enable_llm` flag.
- Uses a pre-trained question-answering model from the Hugging Face `transformers` library (model can be specified via `--llm_model_name`).
- For each article with an abstract, it asks predefined questions to extract:
    - Sample Size
    - Study Design
    - Outcome Type
    - Intervention/Exposure
- Appends these extracted fields to the article's data. If an abstract is missing or an answer isn't found, "Not found" or an error message is stored.

## Output
The system generates a CSV file (e.g., `data/literature_results.csv`). The columns include:
- `Title`
- `Authors` (list of authors, typically represented as a string in the CSV)
- `Journal`
- `DOI`
- `Publication Year`
- `Abstract`
- `Sample Size` (present if LLM preprocessing is enabled)
- `Study Design` (present if LLM preprocessing is enabled)
- `Outcome Type` (present if LLM preprocessing is enabled)
- `Intervention/Exposure` (present if LLM preprocessing is enabled)

## Dependencies
All Python dependencies are listed in `requirements.txt`:
- `biopython`: For NCBI Entrez API interaction.
- `pandas`: For data manipulation and CSV file creation.
- `transformers`: For Hugging Face models (LLM preprocessing).
- `torch`: The deep learning framework used by default for many Hugging Face models. (Or `tensorflow` if models compatible with it are used and TF is preferred). Ensure one is installed if using the LLM feature.
- `logging`: (Standard library) Used for progress and error messages.
```
