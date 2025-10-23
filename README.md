# 🤖 Aeron - AI-Powered Process Analysis Assistant

Aeron is a sophisticated, AI-powered chatbot designed to provide intelligent analysis of business processes and KPIs. It leverages the power of Google's Gemini models and FAISS for efficient similarity search to deliver accurate and context-aware responses. Aeron can also analyze process event logs, generate KPI reports, and provide insights into process bottlenecks and loops.

## ✨ Features

*   **🤖 AI-Powered Chatbot:** A command-line chatbot that can answer questions about business processes, KPIs, and process mining.
*   **🧠 Intelligent Search:** Uses FAISS for efficient similarity search to find the most relevant information from a knowledge base.
*   **🔮 Context-Aware:** Remembers the context of the conversation to provide more accurate and relevant responses.
*   **❓ Question Recommendations:** Suggests related questions to help users explore the knowledge base.
*   **📄 KPI Report Generation:** Generates a complete KPI report package with an executive summary, KPI benchmark, and analysis report.
*   **🔬 Process Analysis:** Analyzes process event logs to identify bottlenecks, loops, and other process inefficiencies.
*   **🔌 Pluggable Architecture:** Easily extendable to support other AI models and knowledge bases.

## 🛠️ Technologies

*   **Python:** The core programming language used for the project.
*   **Google Gemini:** The AI model used for response generation.
*   **FAISS:** A library for efficient similarity search and clustering of dense vectors.
*   **OpenAI:** Used for KPI report generation.
*   **Pandas:** Used for data manipulation and analysis.
*   **Dotenv:** Used for managing environment variables.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Aeron.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up the environment variables:**
    Create a `.env` file in the root directory of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

## Usage

### Chatbot

To start the chatbot, run the following command:

```bash
python main.py
```

The chatbot will load the knowledge base from the `landing.json` file and build a FAISS index. You can then start asking questions about your business processes and KPIs.

### KPI Report Generation

To generate a KPI report, run the following command:

```bash
python report_generator.py
```

The script will use the sample data from the `data.py` file to generate a KPI report package in JSON format.

### Process Analysis

To analyze a process event log, run the following command:

```bash
python y.py
```

The script will use the sample data from the `invoice_process_expanded (1).csv` file to analyze the process and generate a structured process flow output in JSON format.

## 📂 Files

*   **`main.py`**: The main entry point of the chatbot application.
*   **`Landing_page_bot_2.py`**: The core of the chatbot, powered by Gemini and FAISS.
*   **`Landing_page_bot.py`**: An older version of the chatbot, powered by OpenAI's GPT models.
*   **`Invoice_process_function.py`**: Contains functions for processing invoice data from a CSV file.
*   **`report_generator.py`**: Contains a function for generating a KPI report package using OpenAI's API.
*   **`data.py`**: Contains sample data for testing the `report_generator.py` script.
*   **`y.py`**: Contains a function for analyzing and structuring process event logs.
*   **`requirements.txt`**: The project's dependencies.
*   **`landing.json`**: The knowledge base for the chatbot.
*   **`invoice_process_expanded (1).csv`**: Sample data for the process analysis script.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
