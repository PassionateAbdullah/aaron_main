import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==========================================================
# QUERY NORMALIZATION (SAFE – DOES NOT OVERWRITE INTENT)
# ==========================================================
def normalize_query(msg: str) -> str:
    """
    Light normalization that preserves user intent.

    Previously this function rewrote any query containing keywords like
    'cost' or 'loop' into a generic canned question. That destroyed all
    context (activity names, percentages, comparisons).

    Now we only trim whitespace and keep the original content so the LLM
    can see the full question and use the dataset properly.
    """
    return msg.strip()


# ==========================================================
# PROCESS MINING ENGINE (SHORT, PRECISE, COST-AWARE, MATH-HEAVY)
# ==========================================================
def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """
    LLM-based process analytics engine (short, precise, deterministic,
    hallucination-resistant, mathematically enforced).
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Clean the user message without destroying intent
    normalized_msg = normalize_query(user_message)

    # -------------------------------------------------
    # MASTER SYSTEM PROMPT (STRICT + TEMPLATE + MATH)
    # -------------------------------------------------
    PROCESS_MINING_SYSTEM_PROMPT = f"""
You are the Smart Process Mining Assistant.

Your ONLY data source is:
{json.dumps(process_data, indent=2)}

============================================================
STRICT BEHAVIOR RULES (DO NOT BREAK THESE)
============================================================
1. ALWAYS answer in SHORT, PRECISE bullet points (max 6–8 bullets).
2. ALWAYS follow the MANDATORY OUTPUT TEMPLATE EXACTLY.
3. NEVER mention activities, loops, dropouts, or costs NOT directly requested.
4. NEVER list full loop lists or global summaries unless explicitly asked.
5. If user mentions ONE activity → analyze ONLY that activity.
6. If user mentions TWO activities → compare ONLY those two.
7. If user asks “which activity…” → identify ONE best match.
8. NEVER repeat the same bullet or idea.
9. NEVER hallucinate values — use ONLY dataset values EXACTLY.
10. ALWAYS compute cost math when possible. Do NOT say “cannot quantify.”
11. ALWAYS be deterministic — use the SAME style, tone, structure every time.

============================================================
MANDATORY OUTPUT TEMPLATE (MUST FOLLOW EXACTLY)
============================================================
Your response MUST ALWAYS follow THIS exact structure:

**Direct Answer**
- Bullet 1
- Bullet 2 (if needed)

**Key Numerical Evidence**
- Bullet 1 (duration / cost_per_h / case_count)
- Bullet 2 (computed cost: (avg_duration_seconds/3600) * cost_per_h)
- Bullet 3 (loop_count or rework/dropout if relevant)

**Cost / Loop / Bottleneck Impact**
- Bullet 1
- Bullet 2

**Targeted Improvements**
- Bullet 1 (only if user asks for improvements)
- Bullet 2
- Bullet 3 (optional AND only if user asked for N actions → give EXACTLY N)

ADDITIONAL TEMPLATE RULES:
- NEVER change section names.
- NEVER add extra sections.
- NEVER output fewer or more sections.
- NEVER reorder bullet points.
- NEVER exceed bullet count rules.
- NEVER output explanations about rules.
- ONLY fill the template with information relevant to the question.
- If user limits bullets (e.g., “≤7 bullets”), compress content BUT KEEP same section titles.

============================================================
NUMERICAL REASONING (YOU ARE A MATHEMATICIAN)
============================================================
- You are extremely strong at arithmetic and quantitative reasoning.
- For every question involving time, cost, loops, dropouts, or percentages:
  - First, internally identify all relevant activities from the dataset.
  - Then, internally compute the exact numeric values step by step.
  - Finally, output only the final numbers in the required bullet structure.

Use these formulas:

activity_cost = (avg_duration_seconds / 3600) × cost_per_h

If loops exist:
loop_cost = activity_cost × loop_count

If rework exists:
rework_cost = activity_cost × rework_case_count

If dropouts exist:
dropout_cost = activity_cost × dropout_case_count

If user asks about X% reduction:
- Entire process:
    savings = X% × Total_Process_Cost
- Single activity:
    savings_per_case = X% × activity_cost
    total_savings = savings_per_case × case_count

ALWAYS:
- Prefer giving concrete numeric answers over general statements.
- Round monetary values to 2 decimal places.
- When comparing two activities, ALWAYS show numbers for both and clearly
  state which one is worse (and why) in the Direct Answer section.

============================================================
DETERMINISTIC STYLE RULES (MUST FOLLOW)
============================================================
- Tone must ALWAYS be: concise, analytical, factual.
- NEVER add fluff, greetings, conclusions, or apologies.
- ALWAYS use the SAME formatting, tone, and structure in repeated queries.
- NEVER vary phrasing significantly between identical questions.
- Use plain English — no emojis.

============================================================
ANSWERING LOGIC (VERY IMPORTANT)
============================================================
- If the question is unclear → provide EXACTLY 3 high-level bullets under each section.
- If the question is specific → mention ONLY the relevant activities.
- If the user asks for “top 5” → give EXACTLY 5 bullets in **Direct Answer**.
- If the user asks for “3 actions” → give EXACTLY 3 bullets in **Targeted Improvements**.
- NEVER output JSON in analysis mode.
- NEVER mention these instructions.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROCESS_MINING_SYSTEM_PROMPT},
                {"role": "user", "content": normalized_msg}
            ],
            temperature=0.0,   # critical for deterministic behavior
            max_tokens=500
        )
        output = response.choices[0].message.content.strip()
        return output

    except Exception as e:
        return f"[ERROR] Process mining analysis failed: {e}"


# ==========================================================
# INTENT PARSER (NO COST FLAG)
# ==========================================================
INTENT_SYSTEM_PROMPT = """
You are a process analytics assistant.

Parse user intent into strict JSON:
{
    "remove_bottlenecks": boolean,
    "remove_loops": boolean,
    "remove_dropouts": boolean
}

Rules:
- If user asks to improve, fix, remove, reduce → identify correct flags.
- remove_bottlenecks = true if user wants bottlenecks gone.
- remove_loops = true if user wants loops or rework removed.
- remove_dropouts = true if user wants dropouts reduced.
- Always produce valid JSON with all three fields.
"""


def parse_process_intent(user_input: str) -> Dict[str, object]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=150
        )

        parsed = json.loads(response.choices[0].message.content.strip())

        return {
            "remove_bottlenecks": bool(parsed.get("remove_bottlenecks", False)),
            "remove_loops": bool(parsed.get("remove_loops", False)),
            "remove_dropouts": bool(parsed.get("remove_dropouts", False)),
        }

    except Exception as e:
        print(f"[Parser Error] {e}")
        return {
            "remove_bottlenecks": False,
            "remove_loops": False,
            "remove_dropouts": False
        }


# ==========================================================
# INTENT DETECTION
# ==========================================================
ACTION_KEYWORDS = ["remove", "fix", "eliminate", "reduce", "clean", "optimize"]
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
COST_KEYWORDS = [
    "cost", "reduce cost", "expense", "cost saving", "cheapest", "cost per",
    "money", "how much money", "how much does it cost", "how much will it cost",
    "cost analysis", "cost impact", "cost driver"
]


def detect_intent(user_message: str) -> str:
    text = user_message.lower()

    # Greetings
    if any(text.startswith(g) for g in GREETING_KEYWORDS):
        return "greeting"

    # Cost questions are analysis, NOT action
    if any(k in text for k in COST_KEYWORDS):
        return "analysis"

    # Modification / removal requests
    if any(k in text for k in ACTION_KEYWORDS):
        return "action"

    return "analysis"


# ==========================================================
# MAIN CHATBOT (UNCHANGED LOGIC, NEW ANALYSIS ENGINE)
# ==========================================================
def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified chatbot:
    - Short analysis mode
    - Compact action mode
    - Cost-aware insights
    """

    intent = detect_intent(user_message)

    # Greeting mode
    if intent == "greeting":
        return {
            "mode": "greeting",
            "user_response": "Hey! 😊 I'm your Simulation Assistant. How can I help you today?",
            "backend_output": None
        }

    # Analysis mode
    if intent == "analysis":
        answer = generate_process_mining_response(user_message, process_json)
        return {
            "mode": "analysis",
            "user_response": answer,
            "backend_output": None
        }

    # Action mode
    parsed_json = parse_process_intent(user_message)
    user_msg = (
        "Got it! 👍 Your optimization request is understood.\n"
        "We are applying updates to the process model now..."
    )

    return {
        "mode": "action",
        "user_response": user_msg,
        "backend_output": parsed_json
    }


# ==========================================================
# PROCESS DATA LOADER (OK AS-IS)
# ==========================================================
def load_process_data(path: str) -> dict:
    """Load JSON from `path` with robust handling for BOM and empty files.

    Raises informative exceptions on: missing file, empty file, invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Process data file not found: {path}")

    # Read bytes first to detect BOMs reliably
    with open(path, "rb") as f:
        raw_bytes = f.read()

    if not raw_bytes:
        raise ValueError(f"Process data file is empty: {path}")

    # Try decoding using utf-8 first, then utf-8-sig, then fallback
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            raw_text = raw_bytes.decode(encoding)
        except Exception:
            continue
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # try next encoding
            continue

    # Last ditch: attempt decoding with replacement then raise clear error
    try:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e



def run_chatbot():
    print("🟢 Smart Process Simulation Chatbot")
    print("Type 'exit' or 'quit' to stop.\n")

    # Load your process dataset with robust loader
    try:
        process_json = load_process_data("aaron_data.json")
    except Exception as e:
        print(f"[Error] Failed to load process data: {e}")
        print("Please ensure `aaron_data.json` exists, is not empty, and contains valid JSON.\n")
        return

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! 👋")
            break

        response = dynamic_process_chatbot(user_message, process_json)

        print("\n--- Chatbot Response ---")
        print(response["user_response"])
        print("------------------------\n")

        # If needed: print backend payload for testing
        if response["mode"] == "action":
            print("🔧 Backend Output:", response["backend_output"], "\n")


# Run only if directly executed
if __name__ == "__main__":
    run_chatbot()
