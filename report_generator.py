import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()


def generate_complete_kpi_package_openai(
    data: dict,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Generates a complete KPI report package with a single OpenAI API call.

    Returns JSON with top-level keys: "Executive_Summary", "KPI_Benchmark", "Analysis_Report".
    """

    # Initialize client with your OpenAI API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'"))

    # Compact input data for efficient token usage
    compact = json.dumps(data, separators=(",", ":"))

    # --- SYSTEM PROMPT ---
    system_msg = ("""
            You are a senior process intelligence analyst.

            Two datasets are provided:
            - "Current_Project_Data" (Team 1)
            - "Related_Project_Data" (Team 2, the benchmark)

            Return a concise, well-structured JSON report with exactly three top-level keys:
            "Executive_Summary", "KPI_Benchmark", and "Analysis_Report".

            ---

            1. Executive_Summary:
            Write 3–5 sentences comparing Team 1 and Team 2. Clearly explain which team performs better and why, combining observation, interpretation, and summary into one cohesive paragraph.

            ---

            2. KPI_Benchmark:
            Produce an array of objects with the exact following structure:
            {
            "Metric": "<name>",
            "Team_1_Label": take the value of the "(team)(department)" KPI_DATA.Metadata" output format - team(department)",
            "Team_2_Label": take the value of the "(team)(department)" KPI_DATA.Metadata output format - team(department)",
            "Team_1_Value": <value>,
            "Team_2_Value": <value>,
            "Status": "<Team 1 higher by X.X% | Team 2 higher by X.X% | Equal | Team X took Y more <units>>"
            }}

            Rules:
            - All key names must match exactly as shown.
            - Use department names from KPI_DATA.Metadata for Team_1_Label and Team_2_Label.
            - Compute comparisons relative to the lower team’s value:
            • If Team1 > Team2: X.X = ((Team1 - Team2) / Team2) * 100 → "Team 1 higher by X.X%"
            • If Team2 > Team1: X.X = ((Team2 - Team1) / Team1) * 100 → "Team 2 higher by X.X%"
            • If equal: "Equal"
            - Round X.X to one decimal place and include the '%' sign.

            Formatting by metric type:
            • **Proportion / Ratio Metrics** (include "Rate", "Ratio", "First Pass Rate", "First Pass Yield", "FPY", "Process Efficiency Ratio"):
            - Convert values between 0–1 to percentages (×100).
            - Display both team values as percentages with one decimal place.
            - Use percentage-based comparison for Status.
            • **Time / Duration Metrics** (include "time", "duration", "waiting", or units like "hours", "days", "minutes"):
            - Do not use percentages for Status.
            - Compute absolute difference: Y = |Team1 - Team2|, rounded to one decimal.
            - Format Status as: "Team X took Y more <units>" (units inferred from the metric or KPI data).
            • **Other Numeric Metrics**:
            - Keep numeric values as-is (no unit conversion).
            - Use absolute delta wording if percentage difference is not meaningful.

            Ensure:
            - Exactly one Status per metric (no duplicates).
            - Consistent rounding and formatting.
            - No extra commentary in output.

            Metric Type Reference:
            - Average Cycle Time → time (hours; use KPI data unit if available)
            - Idle Time Ratio → proportion (%)
            - Dropout Rate → proportion (%)
            - First Pass Rate → proportion (%)
            - Bottleneck Duration → time (hours)
            - Time Lost to Bottleneck → time (hours)

            ---

            3. Analysis_Report:
            Return a nested object with the following keys, each containing 2–4 sentences that combine evaluation, observation, interpretation, and recommendations:
            {
            "loop_analysis": { "loop_analysis": "..." },
            "bottleneck_analysis": { "bottleneck_analysis": "..." },
            "dropout_analysis": { "dropout_analysis": "..." },
            "happy_path": { "happy_path": "..." },
            "recommendation_to_action": { "recommendation_to_action": "..." },
            "method_notes": { "method_notes": "..." },
            "appendix": { "appendix": "[Brief supporting notes: metric definitions, calculation formulas, assumptions, data caveats/coverage, thresholds or parameters used, and any references to source fields.]" }
            }

            ---

            Output Rules:
            - Return **valid JSON only** (no markdown, no commentary).
            - Preserve numeric meaning.
            - Use "N/A" only when data is unavailable.
            - The Appendix must never be "N/A" — always include at least minimal supporting notes.
            """)


    # --- USER PROMPT ---
    user_msg = f"KPI_DATA:{compact}"

    # --- SINGLE OPENAI CALL ---
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    # Extract model output safely
    text = (
        (getattr(response.choices[0].message, "content", "")
         if hasattr(response.choices[0], "message") else None)
        or getattr(response.choices[0], "text", None)
        or ""
    ).strip()

    if not text:
        raise ValueError("Empty response from OpenAI (KPI package).")

    # Robust JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
        raise ValueError(f"Failed to parse valid JSON from OpenAI output:\n{text}")


# ===================== TEST USAGE =====================
if __name__ == "__main__":
    from data import test_data  # Must contain valid KPI dataset

    print("Generating complete KPI package (OpenAI)...\n")
    result = generate_complete_kpi_package_openai(test_data)
    print(json.dumps(result, indent=4))

