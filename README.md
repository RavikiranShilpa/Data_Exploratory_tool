# Data Explorer Tool

![Data Explorer Tool Logo](<URL TO LOGO OR IMAGE>)

## Overview

The Data Explorer Tool is a Python application designed to assist users in exploring, cleaning, and analyzing datasets. It provides interactive functionalities for data cleaning and exploration in one tab, and data analysis and prediction in another tab.

## Features

- **Tab 1: Data Cleaning and Exploration**
  - Upload and clean datasets using a variety of operations.
  - View basic statistics about the cleaned data.
  - Perform actions such as removing nulls, duplicates, and outliers, filling missing values, and more.
  - the ask the data tab here uses openAI and hence add the key in the code and run the code.

- **Tab 2: Data Analysis and Prediction**
  - Select from available datasets (Agriculture, Weather, Economy, Merged).
  - Ask questions about the selected dataset and receive answers(uses openAI)

## Prerequisites

Before running the tool, ensure you have the following dependencies installed:

- pandas
- requests
- nl2query
- pandasql
- matplotlib
- seaborn
- streamlit
- spacy
- transformers
- openai (if using the commented-out OpenAI code)

  Here's a simple set of instructions in a single section with bullet points for executing the Data Explorer Tool:

markdown
Copy code
# Data Explorer Tool

![Data Explorer Tool Logo](<URL TO LOGO OR IMAGE>)

...

## How to Run

-# Data Explorer Tool

![Data Explorer Tool Logo](<URL TO LOGO OR IMAGE>)

## How to Run

# Data Explorer Tool

![Data Explorer Tool Logo](<URL TO LOGO OR IMAGE>)

## How to Run

1. **Clone the Repository:**
    ```bash
    git clone <REPOSITORY URL>
    cd Data-Explorer-Tool
    ```
    Replace `<REPOSITORY URL>` with the actual URL of your GitHub repository.

2. **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application:**
    ```bash
    streamlit run data_explorer_tool.py
    ```
    This will start the Streamlit development server and open the application in your default web browser.

5. **Interact with the Application:**
    - Open the provided URL (usually http://localhost:8501) in your web browser to access the Data Explorer Tool.
    - Explore features, upload datasets, and ask questions about the data.

### Important Notes:

- **Security Warning:** Streamlit may display a warning regarding security risks for file uploads. Click "Run Anyway" to proceed; this is standard for local development.

- **Port Number:** By default, Streamlit runs on port 8501. If in use, specify a different port using `--port`, like `streamlit run data_explorer_tool.py --port 8765`.

If you encounter issues, check the terminal for error messages. For questions or assistance, feel free to ask!
