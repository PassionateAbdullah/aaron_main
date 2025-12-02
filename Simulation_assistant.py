# ==========================================================
# SMART PROCESS MINING CHATBOT — FINAL MASTER ARCHITECTURE
# ----------------------------------------------------------
# Features:
# - LLM-powered text normalization
# - LLM-powered intent detection (no keywords!)
# - Mathematical dataset-grounded analysis engine
# - Action-mode intent parser
# - Clean UX (user_response vs backend_output)
# - Safe JSON loader
# - CLI interface
# ==========================================================

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ==========================================================
# 1. LLM TEXT NORMALIZER
# ==========================================================
NORMALIZER_PROMPT = """
You are a text normalizer.

Rewrite the user’s message into a clean, clear, explicit version
WITHOUT changing the meaning.

Rules:
- Keep the meaning exactly the same.
- Remove filler phrases (e.g., "uhh", "please", "can you maybe")
- Expand vague commands into explicit instructions.
- Make the message direct and unambiguous.
- Keep all technical intent.

Return ONLY the rewritten message.
"""

def normalize_query(msg: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": NORMALIZER_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return msg.strip()



# ==========================================================
# 2. LLM INTENT CLASSIFIER
# ==========================================================
INTENT_CLASSIFIER_PROMPT = """
You are an intent classifier for a Process Mining Assistant.

Classify the user's message into EXACTLY one of:

1. "greeting"
   - The user is greeting or starting a conversation.

2. "action"
   - The user wants to MODIFY the process model:
       * remove loops
       * fix or eliminate bottlenecks
       * reduce dropouts
       * clean rework / inefficiencies
       * structurally optimize the process

3. "analysis"
   - The user wants insights:
       * cost analysis
       * bottleneck explanation
       * comparisons
       * root-cause analysis
       * performance questions
       * impact calculation
   - WITHOUT modifying the structure.

Return ONLY:
"greeting"
"action"
"analysis"
"""

def detect_intent(normalized_message: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                {"role": "user", "content": normalized_message}
            ],
            temperature=0.0,
            max_tokens=10
        )

        label = response.choices[0].message.content.strip().lower()

        if label not in ["greeting", "action", "analysis"]:
            return "analysis"

        return label

    except Exception:
        return "analysis"



# ==========================================================
# 3. PROCESS MINING ANALYSIS ENGINE
# ==========================================================
def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """
    Mathematical, dataset-bounded, strict process-mining engine.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PROCESS_MINING_SYSTEM_PROMPT = f"""
You are the **Mathematical Process Mining Engine**.

Your ONLY dataset:
{json.dumps(process_data, indent=2)}

========================================================
RULES
========================================================
- Always use dataset numbers only.
- No hallucination.
- No assumptions.
- Maximum 6–8 bullets.
- Bullets must be short, analytical, numeric when needed.

========================================================
MANDATORY MATH LOGIC
========================================================
Let:
  D = avg_duration_seconds
  C = cost_per_h
  Loops = sum of backward loop frequencies
  Drop = dropout_case_count
  Rework = rework_case_count

Compute:
  ActivityCost = (D / 3600) × C
  LoopCost = ActivityCost × Loops
  DropoutWaste = ActivityCost × Drop
  ReworkWaste = ActivityCost × Rework

Root-cause analysis must reference:
  - duration
  - cost
  - loop frequency
  - dropout frequency
  - upstream vs downstream structure
  - bottleneck flags

Never exceed dataset total process cost: 15369.17 USD.

========================================================
OUTPUT FORMAT
========================================================
- 6–8 bullets
- No formulas (only results)
- Use dataset-backed logic
- Directly answer user’s question
========================================================
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROCESS_MINING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] Analysis failed: {e}"



# ==========================================================
# 4. ACTION MODE — PARSE USER'S STRUCTURAL COMMANDS
# ==========================================================
INTENT_SYSTEM_PROMPT = """
Parse structural optimization requests into strict JSON:

{
  "remove_bottlenecks": boolean,
  "remove_loops": boolean,
  "remove_dropouts": boolean
}

Rules:
- If user instructs to clean/remove/fix/update/optimize loops, bottlenecks, or dropouts → TRUE.
- Always return all three fields.
- Only return these fields.
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
# 5. MAIN CHATBOT CONTROLLER
# ==========================================================
def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]):
    normalized = normalize_query(user_message)
    intent = detect_intent(normalized)

    if intent == "greeting":
        return "Hey! 😊 I'm your Simulation Assistant. How can I help you today?"

    if intent == "analysis":
        return generate_process_mining_response(normalized, process_json)

    # action mode
    return parse_process_intent(normalized)



# ==========================================================
# 6. LOAD JSON DATASET SAFELY
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
# 7. CLI APPLICATION
# ==========================================================
def run_chatbot():
    print("🟢 Smart Process Mining Chatbot")
    print("Type 'exit' to quit.\n")

    try:
        process_json = load_process_data("aaron_data.json")
    except Exception as e:
        print("[Error loading dataset]:", e)
        return

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        response = dynamic_process_chatbot(user_message, process_json)

        print("\n--- Chatbot Response ---")
        print(response)
        print("------------------------\n")



# ==========================================================
# MAIN RUNNER
# ==========================================================
if __name__ == "__main__":
    run_chatbot()
