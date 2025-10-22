import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as api_exceptions

load_dotenv()  # Load environment variables from .env file if present

# ========== CONFIGURATION / DEBUG KEY LOADING ==========
# Try to load and normalize the API key; strip surrounding quotes if present.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # try again with stripping (handles ".env" keys wrapped in quotes)
    raw = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_API_KEY = raw.strip().strip('"').strip("'")

def _mask_key(k: str) -> str:
    if not k:
        return "None"
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}{'*'*(len(k)-8)}{k[-4:]}"

# Debug: prints presence, masked value and length (never prints full key)
print(f"DEBUG: GOOGLE_API_KEY present={bool(GOOGLE_API_KEY)} masked={_mask_key(GOOGLE_API_KEY)} length={len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0}")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing Google API key. Please set GOOGLE_API_KEY as env var or in code.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)


# ========== CORE FUNCTION ==========
def generate_team_kpi_analysis_gemini(data: dict, model_name: str = "gemini-2.5-flash") -> dict:
    """

Optimized Gemini call for KPI analysis.
- Merges observation, interpretation, and recommendation into one concise paragraph per section.
- Output must match the exact JSON format, with nested key names.
- Keeps all key insights, no loss of relevance.
- Limits each section to 2 informative lines.
- Ensures JSON-only, data-driven output. 


    """

    compact_data = json.dumps(data, separators=(",", ":"))

    prompt = (
    "You are a senior process intelligence analyst. "
    "Using the given KPI comparison data between two teams, create a concise, insight-rich benchmark analysis. "
    "Your response must be ONLY valid JSON (no markdown, no text outside JSON). "
    "Include exactly these sections: "
    "loop_analysis, bottleneck_analysis, dropout_analysis, top_5_process_variants, "
    "happy_path, recommendation_to_action, method_notes, appendix. "
    "For each section, output the field name as the key (e.g., 'loop_analysis'). "
    "Within each section, provide a single, concise paragraph of 2–4 sentences that merges the observation, interpretation, and recommendation. "
    "Focus on meaning, trends, and actionable insights. Avoid unnecessary context or filler. "
    "Ensure no important details or relationships from the KPI data are lost. "
    "Format the output exactly as shown below, with no extra explanations or markdown:"
    "\n"
    "Expected Output Format:\n"
    "{\n"
    "  \"loop_analysis\": {\n"
    "    \"loop_analysis\": \"[Compact paragraph with merged insights for loop analysis.]\"\n"
    "  },\n"
    "  \"bottleneck_analysis\": {\n"
    "    \"bottleneck_analysis\": \"[Compact paragraph with merged insights for bottleneck analysis.]\"\n"
    "  },\n"
    "  \"dropout_analysis\": {\n"
    "    \"dropout_analysis\": \"[Compact paragraph with merged insights for dropout analysis.]\"\n"
    "  },\n"
    "  \"happy_path\": {\n"
    "    \"happy_path\": \"[Compact paragraph with merged insights for happy path.]\"\n"
    "  },\n"
    "  \"recommendation_to_action\": {\n"
    "    \"recommendation_to_action\": \"[Compact paragraph with merged insights for recommendation.]\"\n"
    "  }\n"
    "  \"method_notes\": {\n"
    "    \"method_notes\": \"[Compact paragraph with merged insights for method notes.]\"\n"
    "  }\n"
    "  \"appendix\": {\n"
    "    \"appendix\": \"[Compact paragraph with merged insights for appendix.]\"\n"
    "  }\n"
    "}"
    f"KPI_DATA:{compact_data}"


)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
    except api_exceptions.NotFound as e:
        print("ERROR: The model you requested is not available for this API/version.")
        try:
            print("Fetching available models...")
            available = genai.list_models()
            print("Available models (sample):")
            # Attempt to print a concise list
            if isinstance(available, (list, tuple)):
                for m in available[:20]:
                    print("-", getattr(m, "name", m))
            elif isinstance(available, dict):
                for k in list(available.keys())[:20]:
                    print("-", k)
            else:
                print(available)
        except Exception:
            print("Could not list models programmatically. Check Google Cloud Console -> Generative AI API -> Models.")
        raise RuntimeError(
            f"Model '{model_name}' not found or unsupported for generate_content. "
            "Try 'gemini-1.5-pro' or run ListModels to see supported models."
        ) from e
    except Exception:
        raise

    # ---- Extract raw output ----
    content = ""
    if hasattr(response, "text") and response.text:
        content = response.text.strip()
    elif hasattr(response, "candidates") and response.candidates:
        try:
            content = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            pass

    # Debug: show raw output

    if not content:
        raise ValueError("Empty response from Gemini. Try using 'gemini-1.5-pro' instead of 'flash'.")

    # ---- Try parsing JSON safely ----
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end])
        else:
            raise ValueError(f"❌ Failed to parse JSON from model output:\n{content}")

    return parsed

# ========== EXECUTIVE SUMMARY ONLY (GEMINI) ==========
def generate_executive_summary_only_gemini(data: dict, model_name: str = "gemini-2.5-flash") -> dict:
    """Return only an executive summary as JSON with key 'Executive_Summary'.

    The content is a precise, informative overview comparing two teams' KPI performance,
    highlighting 2-3 key strengths, 2-3 gaps, and 1-2 immediate actions.
    """

    compact_data = json.dumps(data, separators=(",", ":"))
    prompt = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Return exactly one top-level key named 'Executive_Summary' whose value is a concise paragraph (3-5 sentences). "
        "The paragraph must summarize a comparison between two teams' KPI performance, covering strengths, gaps, and next actions.\n"
        f"KPI_DATA:{compact_data}"
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    content = getattr(response, "text", "") or ""
    content = content.strip()
    if not content and hasattr(response, "candidates") and response.candidates:
        try:
            content = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            content = ""
    if not content:
        raise ValueError("Empty response from Gemini (executive summary only).")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        raise ValueError(f"Failed to parse Executive_Summary JSON from model output:\n{content}")


# ========== KPI BENCHMARK TABLE (GEMINI) ==========
def generate_kpi_benchmark_table_gemini(data: dict, model_name: str = "gemini-2.5-flash") -> dict:
    """Generate a KPI benchmark table with the exact requested JSON format.

    Output format:
    {
      "KPI_Benchmark": [
        {"Metric": "Process Efficiency", "Current Value": "72%", "Target Value": "85%", "Status": "Below Target"},
        ...
      ]
    }
    Values should be derived from the provided data where available; otherwise infer sensible targets and statuses.
    Keep keys and casing EXACTLY as shown.
    """

    compact_data = json.dumps(data, separators=(",", ":"))
    prompt = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Return exactly one top-level key: 'KPI_Benchmark' which maps to an array of rows. "
        "Each row must contain EXACT keys with this casing: 'Metric', 'Current Value', 'Target Value', 'Status'. "
        "Prefer common metrics when relevant: Process Efficiency, Cycle Time, Error Rate, Customer Satisfaction, Cost per Transaction. "
        "Format values human-readably (e.g., percentages 72%, durations 4.2 days, currency $12.50). "
        "Derive values from the data if present; otherwise infer reasonable targets and evaluate status (e.g., Good, Below Target, Needs Improvement, Above Target, High, Low).\n"
        f"KPI_DATA:{compact_data}"
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    content = getattr(response, "text", "") or ""
    content = content.strip()
    if not content and hasattr(response, "candidates") and response.candidates:
        try:
            content = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            content = ""
    if not content:
        raise ValueError("Empty response from Gemini (KPI benchmark table).")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        raise ValueError(f"Failed to parse KPI_Benchmark JSON from model output:\n{content}")


# ========== WRAPPER: EXEC SUMMARY + KPI TABLE + ANALYSIS ==========
def generate_complete_kpi_package_gemini(
    data: dict,
    summary_model_name: str = "gemini-2.5-flash",
    benchmark_model_name: str = "gemini-2.5-flash",
    report_model_name: str = "gemini-2.5-flash",
) -> dict:
    """Return three sections as one JSON: Executive Summary, KPI Benchmark, Analysis Report.

    Returns a dict with the exact top-level keys and shapes expected by the UI:
    {
        "Executive_Summary": str,
        "KPI_Benchmark": list[dict],
        "Analysis_Report": dict
    }
    """
    summary = generate_executive_summary_only_gemini(data, model_name=summary_model_name)
    benchmark = generate_kpi_benchmark_table_gemini(data, model_name=benchmark_model_name)
    analysis = generate_team_kpi_analysis_gemini(data, model_name=report_model_name)
    return {
        "Executive_Summary": summary.get("Executive_Summary", ""),
        "KPI_Benchmark": benchmark.get("KPI_Benchmark", []),
        "Analysis_Report": analysis,
    }


# ========== PUBLIC FACADE (for backend use) ==========
def generate_kpi_package(
    data: dict,
    *,
    summary_model_name: str = "gemini-2.5-flash",
    benchmark_model_name: str = "gemini-2.5-flash",
    report_model_name: str = "gemini-2.5-flash",
) -> dict:
    """Backend-friendly facade that accepts dynamic JSON and returns:
    {
        "Executive_Summary": "...",
        "KPI_Benchmark": [...],
        "Analysis_Report": {...}
    }
    """
    return generate_complete_kpi_package_gemini(
        data,
        summary_model_name=summary_model_name,
        benchmark_model_name=benchmark_model_name,
        report_model_name=report_model_name,
    )

# ========== WRAPPER: SUMMARY + REPORT ==========
def generate_full_kpi_report_with_summary_gemini(
    data: dict,
    summary_model_name: str = "gemini-2.5-flash",
    report_model_name: str = "gemini-2.5-flash",
) -> dict:
    """Return combined executive summary and detailed report as JSON.

    Shape:
    {
        "executive_summary": {...},
        "report": {...}
    }
    """
    summary = generate_executive_summary_only_gemini(data, model_name=summary_model_name)
    benchmark = generate_kpi_benchmark_table_gemini(data, model_name=benchmark_model_name)
    analysis = generate_team_kpi_analysis_gemini(data, model_name=report_model_name)
    return {"executive_summary": summary,"KPI_Benchmark_table": benchmark, "Report Analysis": analysis}

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Import sample data only for CLI usage to avoid import-time errors in backend
    try:
        from data import test_data  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import test_data from data.py for CLI demo. "
            "Run this module via your backend with real JSON data or fix data.py."
        ) from e

    combined = generate_complete_kpi_package_gemini(test_data)

    print("================ EXEC SUMMARY + KPI TABLE + ANALYSIS ================\n")
    print(json.dumps(combined, indent=4))
    print("\n=====================================================================")
