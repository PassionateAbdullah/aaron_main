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
        "Two datasets are provided: 'Current_Project_Data' (Team 1) and 'Related_Project_Data' (Team 2, the target/benchmark). "
        "Return a concise, well-structured JSON report with exactly three top-level keys: "
        "'Executive_Summary', 'KPI_Benchmark', and 'Analysis_Report'.\n\n"

        "1. Executive_Summary: 3–5 sentences. Provide a comparative evaluation between Team 1 and Team 2: "
        "which team performs better and why. Combine observation, interpretation, and summary into one coherent paragraph.\n\n"



        "2. KPI_Benchmark: An array of objects with these exact fields: \n"
        "   {\n"
        "     \"Metric\": \"<name>\",\n"
        "     \"Team 1 (<Dept from KPI_DATA.Metadata>)\": <value>,\n"
        "     \"Team 2 (<Dept from KPI_DATA.Metadata>)\": <value>,\n"
        "     \"Status (Team 1 vs Team 2)\": \"<Team 1 - X.X% Above Target|Team 1 - X.X% Below Target|Equal>\"\n"
        "   }\n"
        "   Rules: Use department names from KPI_DATA.Metadata for the headers. Compute percentage as ((Team1 - Team2)/Team2)*100, "
        "round to 1 decimal, include the % sign, and explicitly name which team is above/below: if Team1>Team2 use 'Team 1 - X.X% Above Target'; if Team1<Team2 use 'Team 1 - X.X% Below Target'; if equal use 'Equal'. "
        "For metrics that represent proportions (names include 'Rate' or 'Ratio', e.g., 'Dropout Rate', 'First Pass Rate', 'First Pass Yield', 'FPY', 'Process Efficiency Ratio'), format Team 1 and Team 2 values as percentages with the % symbol. "
        "If such values are provided as decimals between 0 and 1 (e.g., 0.49), convert to percentages (49.0%) with 1 decimal place. For other metrics, keep numeric values without units.\n\n"



        "3. Analysis_Report: A nested object with the following keys and 2–4 sentences each, combining evaluation,observation, interpretation, and recommendations: \n"
        "{\n"
        "  \"loop_analysis\": { \"loop_analysis\": \"...\" },\n"
        "  \"bottleneck_analysis\": { \"bottleneck_analysis\": \"...\" },\n"
        "  \"dropout_analysis\": { \"dropout_analysis\": \"...\" },\n"
        "  \"happy_path\": { \"happy_path\": \"...\" },\n"
        "  \"recommendation_to_action\": { \"recommendation_to_action\": \"...\" },\n"
        "  \"method_notes\": { \"method_notes\": \"...\" },\n"
        "  \"appendix\": { \"appendix\": \"[Brief supporting notes: metric definitions, calculation formulas, assumptions, data caveats/coverage, thresholds or parameters used, and any references to source fields.]\" }\n"
        "}\n\n"

        "Output rules: valid JSON only (no markdown), preserve numeric meaning, no extra commentary. Use 'N/A' only when data is unavailable; however, the Appendix must not be 'N/A'—always include brief supporting notes (even if minimal)."
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