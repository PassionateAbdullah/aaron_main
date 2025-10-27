import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()

def generate_complete_kpi_package_openai(
    data: dict,
    model_name: str = "gpt-4o-mini"
) -> dict:
    """
    Generates a complete KPI report package with a single OpenAI API call.

    This function uses one optimized prompt to instruct the LLM to produce a fully
    structured JSON output with three top-level keys:

    {
        "Executive_Summary": "...",
        "KPI_Benchmark": [...],
        "Analysis_Report": {...}
    }

    - Executive_Summary: a 3–5 sentence overview of team performance, strengths, and improvements.
    - KPI_Benchmark: a list of metrics with fields "Metric", "Current Value", "Target Value", and "Status".
    - Analysis_Report: an object with analytical paragraphs (2–4 sentences) for each process category.
    """

    # Initialize client with your OpenAI API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'"))

    # Compress input data to compact JSON for efficient token usage
    compact = json.dumps(data, separators=(",", ":"))

    # --- SYSTEM PROMPT: Guides LLM to output structured business insight JSON ---
    system_msg = (
    
        "You are a senior process intelligence analyst. "
        "You are provided KPI data for two teams:\n"
        " - 'Current_Project_Data' represents the team being analyzed.\n"
        " - 'Related_Project_Data' represents the benchmark or target team.\n\n"
        "Generate a comprehensive yet concise comparative process intelligence report "
        "between these two datasets. The report must contain exactly three top-level keys:\n\n"
        "1. 'Executive_Summary': A 4–6 sentence overview summarizing performance trends, "
        "strengths, weaknesses, and actionable takeaways comparing both teams.\n\n"
        "2. 'KPI_Benchmark': An array comparing critical KPIs side by side with the following fields:\n"
        "   - 'Metric': Name of the KPI metric\n"
        "   - 'Current Value': Value from Current_Project_Data\n"
        "   - 'Target Value': Value from Related_Project_Data\n"
        "   - 'Status': One of 'Above Target', 'Below Target', or 'On Target'.\n"
        "   Interpret percentages, durations, and ratios appropriately.\n\n"
        "3. 'Analysis_Report': A nested JSON object that contains deeper analytical sections.\n"
        "   Each section must be structured exactly as shown below:\n\n"
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
        "  },\n"
        "  \"method_notes\": {\n"
        "    \"method_notes\": \"[Compact paragraph with merged insights for method notes.]\"\n"
        "  },\n"
        "  \"appendix\": {\n"
        "    \"appendix\": \"[Compact paragraph with merged insights for appendix.]\"\n"
        "  }\n"
        "}\n\n"
        "Each paragraph (2–4 sentences) should combine observation, interpretation, and recommendation concisely.\n\n"
        "Rules:\n"
        "- Output must be **valid JSON only** (no markdown or text outside JSON).\n"
        "- Ensure analytical meaning and numeric comparisons are preserved.\n"
        "- No extra commentary or formatting.\n"
        "- Include 'N/A' only when data is unavailable.\n\n"
        f"KPI_DATA: {compact}"
    )

    # --- USER PROMPT: Includes KPI data context ---
    user_msg = f"KPI_DATA:{compact}"

    # --- SINGLE OPENAI CALL ---
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,     # Lower temperature for factual, consistent outputs
        max_tokens=2000,     # Enough room for large reports
    )

    # Extract model output safely
    text = (
        (getattr(response.choices[0].message, "content", "")
         if hasattr(response.choices[0], "message") else None)
        or getattr(response.choices[0], "text", None)
        or ""
    ).strip()

    # Check for empty output
    if not text:
        raise ValueError("Empty response from OpenAI (KPI package).")

    # --- Robust JSON parsing ---
    try:
        # Try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # If LLM adds stray text, trim to JSON boundaries
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise ValueError(f"❌ Failed to parse valid JSON from OpenAI output:\n{text}")


# ===================== TEST USAGE =====================
if __name__ == "__main__":
    from data import test_data  # Must contain valid KPI dataset

    print("🧠 Generating complete KPI package (OpenAI)...\n")
    result = generate_complete_kpi_package_openai(test_data)
    print(json.dumps(result, indent=4))