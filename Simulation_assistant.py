# ==========================================================
# SMART PROCESS MINING CHATBOT — MASTER ARCHITECTURE
# Single-file production-ready version
# ==========================================================

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables
load_dotenv()


# ==========================================================
# NORMALIZATION (Minimal, clean)
# ==========================================================
def normalize_query(msg: str) -> str:
    """Return user text without altering meaning."""
    return msg.strip()


# ==========================================================
# ULTRA-STRONG PROCESS MINING ENGINE
# ==========================================================
def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """
    Mathematical, dataset-bounded, zero-hallucination
    process mining analysis engine.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PROCESS_MINING_SYSTEM_PROMPT = f"""
You are the **Mathematical Process Mining Engine**.

Your ONLY dataset:
{json.dumps(process_data, indent=2)}

========================================================
GLOBAL BEHAVIOR
========================================================
You MUST:
- Think like a mathematician
- Analyze like a process-mining expert
- Compare activities precisely
- Use dataset numbers ONLY
- Never hallucinate
- Provide short, dense bullets
- Maximum 6–8 bullets

========================================================
MATH DEFINITIONS (MANDATORY)
========================================================
Let:
- D = avg_duration_seconds
- C = cost_per_h
- Loops = frequencies in loopConnections
- LoopCount = loop_count
- Drop = dropout_case_count
- Rework = rework_case_count
- TotalCount = total_count
- CaseCount = case_count

Use the following COST RULES ALWAYS:

1) ActivityCost = (D / 3600) × C  
2) LoopCost = ActivityCost × (sum of backward loop frequencies)  
3) DropoutWaste = ActivityCost × Drop  
4) ReworkWaste = ActivityCost × Rework  
5) BottleneckImpact increases if:
     • D is significantly above process average OR
     • The activity is upstream in many loops

Variant Cost:
- Sum(activity cost × frequency in variant)

Only compute values using dataset.  
Never exceed dataset's total process cost: 15369.17 USD.

========================================================
RESPONSE FORMAT
========================================================
- 6–8 bullets
- Direct answers only
- Each bullet must reference dataset values
- If question asks "why" → causal explanation
- If question asks "which" → direct comparison
- If question asks "how much" → compute numerically
- No long paragraphs
- No invented numbers

========================================================
DO NOT:
- Invent values
- Use external knowledge
- Assume missing numbers
- Produce long paragraphs
========================================================

Now answer the user's question using ONLY the dataset.
    """

    normalized = normalize_query(user_message)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROCESS_MINING_SYSTEM_PROMPT},
                {"role": "user", "content": normalized}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] Analysis failed: {e}"


# ==========================================================
# ACTION MODE — INTENT PARSER
# ==========================================================

INTENT_SYSTEM_PROMPT = """
Parse optimization requests into strict JSON:

{
  "remove_bottlenecks": boolean,
  "remove_loops": boolean,
  "remove_dropouts": boolean
}

Rules:
- If user says: remove, fix, eliminate, reduce, clean, optimize → TRUE
- Only these 3 flags may exist
- Always return all fields
- Never add extra fields
"""

def parse_process_intent(user_input: str) -> Dict[str, object]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            "remove_dropouts": bool(parsed.get("remove_dropouts", False))
        }

    except Exception:
        return {
            "remove_bottlenecks": False,
            "remove_loops": False,
            "remove_dropouts": False
        }


# ==========================================================
# INTENT DETECTION LOGIC
# ==========================================================
ACTION_KEYWORDS = [
    "remove bottlenecks",
    "remove loops",
    "remove dropouts",
    "fix bottlenecks",
    "fix loops",
    "fix dropouts",
    "eliminate bottlenecks",
    "eliminate loops",
    "eliminate dropouts",
    "clean loops",
    "clean rework",
    "clean inefficiencies",
    "clean dropouts",
    "clean it",
    "clean that",
    "remove the dropout",
    "reduce dropout",
    "reduce the bottlenecks",
    "reduce bottleneck",
    "reduce the loops",
    "reduce loop",
    "optimize bottlenecks",
    "optimize loops",
    "optimize dropouts",
    "optimize it",
    "optimize that",
    "clean bottlenecks",
    
]
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def detect_intent(user_message: str) -> str:
    text = user_message.lower()

    # Greeting mode
    if any(text.startswith(g) for g in GREETING_KEYWORDS):
        return "greeting"

    # Action mode
    if any(k in text for k in ACTION_KEYWORDS):
        return "action"

    # Everything else = analysis mode
    return "analysis"


# ==========================================================
# MAIN CHATBOT CONTROLLER
# ==========================================================
def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]) -> Dict[str, Any]:
    intent = detect_intent(user_message)

    # --------------------------
    # GREETING MODE
    # --------------------------
    if intent == "greeting":
        return {
            "mode": "greeting",
            "user_response": "Hey! 😊 I'm your Simulation Assistant. How can I help you today?",
            "backend_output": None
        }

    # --------------------------
    # ANALYSIS MODE
    # --------------------------
    if intent == "analysis":
        answer = generate_process_mining_response(user_message, process_json)
        return {
            "mode": "analysis",
            "user_response": answer,
            "backend_output": None
        }

    # --------------------------
    # ACTION MODE
    # --------------------------
    parsed_json = parse_process_intent(user_message)

    user_msg = (
        "Got it! 👍 Your optimization request is understood.\n"
        "We are applying updates to the process model now..."
    )

    return {
        "mode": "action",
        "user_response": user_msg,      # what the user sees
        "backend_output": parsed_json   # internal action JSON, separate & clean
    }

# ==========================================================
# SAFE JSON LOADER
# ==========================================================
def load_process_data(path: str) -> dict:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Process data file not found: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    if not raw:
        raise ValueError("File is empty.")

    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            pass

    raise ValueError("Invalid JSON format.")


# ==========================================================
# CLI RUNNER
# ==========================================================
def run_chatbot():
    print("🟢 Smart Process Mining Chatbot")
    print("Type 'exit' to quit.\n")

    try:
        process_json = load_process_data("aaron_data.json")
    except Exception as e:
        print("[Error loading dataset]", e)
        return

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        out = dynamic_process_chatbot(user_message, process_json)

        print("\n--- Chatbot Response ---")
        print(out["user_response"])
        print("------------------------\n")

        if out["mode"] == "action":
            print("🔧 Backend Output:", out["backend_output"], "\n")


# Run if executed directly
if __name__ == "__main__":
    run_chatbot()
