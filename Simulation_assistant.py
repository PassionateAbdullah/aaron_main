import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()


# ==========================================================
# NORMALIZATION HELPERS
# ==========================================================
def normalize_query(msg: str) -> str:
    msg = msg.strip().lower()

    replacement_map = {
        "bottleneck": "where is the bottleneck in the process?",
        "improve": "what can be improved in the process?",
        "help": "explain the process issues and improvements",
        "slow": "which activities are slow and causing delays?",
        "loop": "which steps contain loops or rework?",
        "dropout": "which steps have dropouts?",
        "rework": "which steps have rework cycles?",
        "cost": "cost analysis of the process"
    }

    for key, value in replacement_map.items():
        if key in msg:
            return value
    return msg


# ==========================================================
# PROCESS MINING ENGINE (SHORT, PRECISE, COST-AWARE)
# ==========================================================
def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """
    LLM-based process analytics engine (short, precise, key-point responses).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PROCESS_MINING_SYSTEM_PROMPT = f"""
You are the Smart Process Mining Assistant.

Your ONLY dataset:
{json.dumps(process_data, indent=2)}

------------------------------------------------
RESPONSE RULES
------------------------------------------------
1. ALWAYS respond in SHORT, PRECISE bullet points.
2. Max 6–8 bullets.
3. Highlight ONLY:
   - Key bottlenecks
   - Loops / rework
   - Dropouts
   - Cost drivers + cost impact
   - Improvements (max 3 bullets)
4. No long paragraphs.
5. Never invent data.
6. Use dataset values only.

------------------------------------------------
COST LOGIC RULES
------------------------------------------------
- activity_cost = (avg_duration_seconds / 3600) × cost_per_h
- loops multiply cost because they repeat work
- bottlenecks increase cost when duration is long + cost_per_h is high
- dropouts waste cost spent before case exits
- total process cost comes from dataset (do not fabricate)

------------------------------------------------
ANSWERING STYLE
------------------------------------------------
- Simple question → simple bullets.
- Technical question → technical bullets.
- If unclear → short summary + insights.
    """

    normalized_msg = normalize_query(user_message)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROCESS_MINING_SYSTEM_PROMPT},
                {"role": "user", "content": normalized_msg}
            ],
            temperature=0.2,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()

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
COST_KEYWORDS = ["cost", "reduce cost", "expense", "cost saving", "cheapest", "cost per", "money", "how much money", "how much does it cost", "how much will it cost", "how much money will it cost","how much will it cost", "how much does it cost", "expense", "cost analysis", "cost impact", "cost driver", "cost impact","how much"]


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
# MAIN CHATBOT (FINAL COMBINED FUNCTION)
# ==========================================================
def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified chatbot:
    - Short analysis mode
    - Compact action mode
    - Cost-aware insights
    """

    intent = detect_intent(user_message)

    # -------------------------
    # GREETING MODE
    # -------------------------
    if intent == "greeting":
        return {
            "mode": "greeting",
            "user_response": "Hello! 👋 How can I support your process simulation today?",
            "backend_output": None
        }

    # -------------------------
    # ANALYSIS MODE
    # -------------------------
    if intent == "analysis":
        answer = generate_process_mining_response(user_message, process_json)
        return {
            "mode": "analysis",
            "user_response": answer,
            "backend_output": None
        }

    # -------------------------
    # ACTION MODE (COMPACT)
    # -------------------------
    parsed_json = parse_process_intent(user_message)

    user_msg = (
        "✔ Optimization request registered.\n\n"
        "**Applying updates:**\n"
        f"- Remove bottlenecks: **{parsed_json.get('remove_bottlenecks')}**\n"
        f"- Remove loops: **{parsed_json.get('remove_loops')}**\n"
        f"- Reduce dropouts: **{parsed_json.get('remove_dropouts')}**\n\n"
        "⏳ Updating the process model..."
    )

    return {
        "mode": "action",
        "user_response": user_msg,
        "backend_output": parsed_json
    }


import json
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
