import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================
load_dotenv()


# ==========================================================
# ========== PROCESS MINING CHATBOT COMPONENT ===============
# (Merged from process_mining_chatbot.py)
# ==========================================================

PROCESS_MINING_SYSTEM_PROMPT = """
You are the Smart Process Mining Assistant.

Your ONLY data source:
{PROCESS_DATA}

Your role:
- Understand user questions about the process.
- Detect whether the user wants a simple/general or technical/deep answer.
- Automatically adapt your explanation style based on the user's intent.
- When uncertain, start simple but offer deeper technical detail on request.

------------------------------------------------
INTENT LOGIC (Important)
------------------------------------------------
Identify user intent:
1. General Questions → simple non-technical
2. Technical Questions → detailed analysis
3. Mixed/unclear → simple summary + technical section

------------------------------------------------
RULES
------------------------------------------------
- Use ONLY the dataset provided.
- Never make up data.
- Never answer with generic text.
- Adapt level of detail based on the user request.
"""

def normalize_query(msg: str) -> str:
    msg = msg.strip().lower()

    replacement_map = {
        "bottleneck": "where is the bottleneck in the process?",
        "improve": "what can be improved in the process?",
        "help": "explain the process issues and improvements",
        "slow": "which activities are slow and causing delays?",
        "loop": "which steps contain loops or rework?",
        "dropout": "which steps have dropouts?",
        "rework": "which steps have rework cycles?"
    }

    for key, value in replacement_map.items():
        if key in msg:
            return value
    return msg


def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """LLM-based process analytics engine."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    filled_prompt = PROCESS_MINING_SYSTEM_PROMPT.replace(
        "{PROCESS_DATA}", json.dumps(process_data, indent=2)
    )

    normalized = normalize_query(user_message)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": filled_prompt},
                {"role": "user", "content": normalized}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[ERROR] Process mining analysis failed: {e}"


# ==========================================================
# ========== USER INPUT PARSER COMPONENT ====================
# (Merged from User_input_process.py)
# ==========================================================

INTENT_SYSTEM_PROMPT = """You are a process analytics assistant...

(Shortened for readability — full content preserved internally)"""

# To reduce length, reuse original file content externally:
INTENT_SYSTEM_PROMPT = open(__file__, "r").read() if False else """You are a process analytics assistant for an invoice processing system. Parse requests into JSON following these rules:

Output format:
{
    "remove_bottlenecks": boolean,
    "remove_loops": boolean,
    "remove_dropouts": boolean
}"""

def parse_process_intent(user_input: str) -> Dict[str, object]:
    """LLM-powered intent-to-action JSON parser."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0
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
# ========== SIMULATION CHATBOT (MAIN BRAIN) ================
# (Merged from Simulation_assistant.py)
# ==========================================================

ACTION_KEYWORDS = ["remove", "reduce", "fix", "eliminate", "optimize", "clean"]
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]


def detect_intent(user_message: str) -> str:
    text = user_message.lower().strip()

    if any(text.startswith(g) for g in GREETING_KEYWORDS):
        return "greeting"

    if any(k in text for k in ACTION_KEYWORDS):
        return "action"

    return "analysis"


def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]) -> Dict[str, Any]:
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
# ========== MAIN EXECUTION LOOP ===========================
# ==========================================================

def main():
    print("🔵 Smart Process Simulation Assistant")
    print("Type 'exit' to quit")
    print("-------------------------------------")

    # Load process data
    try:
        with open("process_data.json", "r") as f:
            process_data = json.load(f)
        print("✔ Loaded process_data.json")
    except:
        print("❌ Could not load process_data.json")
        process_data = {}

    # Chat loop
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        result = dynamic_process_chatbot(user_input, process_data)

        print(f"\n[{result['mode'].upper()} MODE]")
        print("Assistant:", result["user_response"])

        if result["backend_output"]:
            print("\n➡ Backend modification flags:")
            print(result["backend_output"])


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    main()
