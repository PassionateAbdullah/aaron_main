import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# ===================== CONFIGURATION =====================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY. Please set it in your .env file or environment.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# ===================== HELPER =====================
def extract_response_text(response):
    """
    Extracts content text safely from OpenAI chat completion responses
    across SDK versions.
    """
    message = getattr(response.choices[0], "message", None)
    if isinstance(message, dict):
        return message.get("content", "").strip()
    elif message and hasattr(message, "content"):
        return message.content.strip()
    # fallback to text if exists
    return getattr(response.choices[0], "text", "").strip()


# ===================== EXECUTIVE SUMMARY =====================
def generate_executive_summary_openai(data: dict, model_name: str = "gpt-4o-mini") -> dict:
    """
    Generates a concise executive summary comparing two teams' KPI performance.
    Returns: {"Executive_Summary": "..."}
    """
    compact = json.dumps(data, separators=(",", ":"))

    system_msg = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Return exactly one top-level key: 'Executive_Summary'. "
        "Provide a 3–5 sentence overview summarizing team performance, key strengths, weaknesses, and improvement areas."
    )

    user_msg = f"KPI_DATA:{compact}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    text = extract_response_text(response)

    if not text:
        raise ValueError("Empty response from OpenAI (Executive Summary).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise ValueError(f"Failed to parse Executive_Summary JSON:\n{text}")


# ===================== KPI BENCHMARK TABLE =====================
def generate_kpi_benchmark_table_openai(data: dict, model_name: str = "gpt-4o-mini") -> dict:
    """
    Generates a KPI benchmark table in structured JSON format.
    Output format:
    {
      "KPI_Benchmark": [
        {"Metric": "Process Efficiency", "Current Value": "72%", "Target Value": "85%", "Status": "Below Target"},
        ...
      ]
    }
    """
    compact = json.dumps(data, separators=(",", ":"))

    system_msg = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Return one key 'KPI_Benchmark' mapping to an array of rows with keys: "
        "'Metric', 'Current Value', 'Target Value', 'Status'. "
        "Use clear, business-relevant values and human-readable formatting (%, $, days, etc.)."
    )

    user_msg = (
        "Prefer metrics like Process Efficiency, Cycle Time, Error Rate, Customer Satisfaction, and Cost per Transaction. "
        f"Base the table on KPI_DATA:{compact}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    text = extract_response_text(response)

    if not text:
        raise ValueError("Empty response from OpenAI (KPI benchmark table).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise ValueError(f"Failed to parse KPI_Benchmark JSON:\n{text}")


# ===================== KPI ANALYSIS REPORT =====================
def generate_team_kpi_analysis_openai(data: dict, model_name: str = "gpt-4o-mini") -> dict:
    """
    Generates a structured KPI benchmark analysis using OpenAI.
    Output is valid JSON with concise sections combining observation, interpretation, and recommendation.
    """
    compact = json.dumps(data, separators=(",", ":"))

    system_msg = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Include exactly these top-level keys, each containing a concise analytical paragraph (2–4 sentences): "
        "loop_analysis, bottleneck_analysis, dropout_analysis, top_5_process_variants, happy_path, "
        "recommendation_to_action, method_notes, appendix. "
        "Each key should contain a short text insight merging observation, interpretation, and recommendation."
    )

    user_msg = f"KPI_DATA:{compact}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    text = extract_response_text(response)

    if not text:
        raise ValueError("Empty response from OpenAI (KPI analysis).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise ValueError(f"Failed to parse JSON from OpenAI output:\n{text}")


# ===================== COMBINED WRAPPER =====================
def generate_complete_kpi_package_openai(
    data: dict,
    summary_model_name: str = "gpt-4o-mini",
    benchmark_model_name: str = "gpt-4o-mini",
    report_model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Generates a complete KPI report package with three sections:
    {
        "Executive_Summary": "...",
        "KPI_Benchmark": [...],
        "Analysis_Report": {...}
    }
    """
    summary = generate_executive_summary_openai(data, model_name=summary_model_name)
    benchmark = generate_kpi_benchmark_table_openai(data, model_name=benchmark_model_name)
    analysis = generate_team_kpi_analysis_openai(data, model_name=report_model_name)

    return {
        "Executive_Summary": summary.get("Executive_Summary", ""),
        "KPI_Benchmark": benchmark.get("KPI_Benchmark", []),
        "Analysis_Report": analysis,
    }


# ===================== TEST ENTRY POINT =====================
if __name__ == "__main__":
    try:
        from data import test_data  # make sure you have data.py with test_data
    except ImportError:
        raise RuntimeError("❌ Missing test_data in data.py. Please create a sample KPI JSON input.")

    print("Generating KPI Analysis Report (OpenAI)...\n")
    result = generate_complete_kpi_package_openai(test_data)
    print(json.dumps(result, indent=4))
