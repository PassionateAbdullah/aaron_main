# Aeron - Data Analysis and Chatbot Project

This project, named "Aeron", is a multi-faceted tool that combines data analysis of invoice processing and Key Performance Indicators (KPIs) with a chatbot interface for interacting with the data.

## Functionality

The project has two main functionalities:

1.  **Invoice and KPI Analysis:**
    *   Processes invoice data from a CSV file (`invoice_process_expanded (1).csv`).
    *   Analyzes Key Performance Indicators (KPIs) of different teams based on sample data (`data.py`).
    *   Utilizes the Gemini API to generate a detailed KPI analysis report, which includes:
        *   Loop analysis
        *   Bottleneck analysis
        *   Dropout analysis
        *   Recommendations for action

2.  **Chatbot:**
    *   Provides a command-line interface (CLI) chatbot to interact with the KPI data and analysis.
    *   The chatbot uses a knowledge base (`landing.json`) and the Gemini API to answer user questions.
    *   It leverages the FAISS library for efficient similarity search to find the most relevant answers in the knowledge base.

## Project Structure

```
├──.gitignore
├──data.py
├──invoice_process_expanded (1).csv
├──Landing_page_bot_2.py
├──Landing_page_bot.py
├──landing.json
├──main.py
├──report_generator.py
├──requirements.txt
├──y.py
├──__pycache__
│  ├──data.cpython-311.pyc
│  └──Landing_page_bot_2.cpython-311.pyc
├──.git
│  ├──hooks
│  ├──info
│  ├──logs
│  ├──objects
│  └──refs
└──.venv
   ├──Include
   ├──Lib
   ├──pyvenv.cfg
   └──Scripts
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your Google API Key:**
    *   Create a `.env` file in the root directory of the project.
    *   Add your Google API key to the `.env` file as follows:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY"
        ```

## Usage

### Chatbot

To start the chatbot, run the `main.py` file:

```bash
python main.py
```

The chatbot will load the knowledge base and you can start asking questions.

### Report Generation

To generate the KPI analysis report, run the `report_generator.py` file:

```bash
python report_generator.py
```

The script will process the data from `data.py` and output a JSON-formatted report to the console.
