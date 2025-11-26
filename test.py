from typing import Dict, Any
import json
from process_mining_engine import generate_process_mining_response
from intent_parser import parse_process_intent

ACTION_KEYWORDS = ["remove", "reduce", "fix", "optimize", "eliminate", "clean"]
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

    # Default backend flags for all modes
    default_backend_flags = {
        "remove_bottlenecks": False,
        "remove_loops": False,
        "remove_dropouts": False
    }

    # ==========================================================
    # 🟢 GREETING MODE
    # ==========================================================
    if intent == "greeting":
        return {
            "mode": "greeting",
            "user_response": "Hey! 😊 I'm your Simulation Assistant. How can I help you today?",
            "backend_output": default_backend_flags
        }

    # ==========================================================
    # 🔵 ANALYSIS MODE
    # ==========================================================
    if intent == "analysis":
        user_answer = generate_process_mining_response(user_message, process_json)

        return {
            "mode": "analysis",
            "user_response": user_answer,
            "backend_output": default_backend_flags
        }

    # ==========================================================
    # 🟠 ACTION MODE
    # ==========================================================
    if intent == "action":
        try:
            parsed_flags = parse_process_intent(user_message)
        except Exception:
            parsed_flags = default_backend_flags

        user_msg = (
            "Got it! 👍 Your optimization request is understood.\n"
            "We are applying updates to the process model now..."
        )

        return {
            "mode": "action",
            "user_response": user_msg,
            "backend_output": parsed_flags
        }


def load_process_data(path="process_data.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        print("⚠️ Failed to load process_data.json")
        return {}

def main():
    print("🔵 Smart Process Simulation Assistant")
    print("Type 'exit' to quit")
    print("-------------------------------------")

    process_data = load_process_data()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        result = simulation_chatbot(user_input, process_data)

        print(f"\n[{result['mode'].upper()} MODE]")
        print("Assistant:", result["user_response"])

        if result["backend_output"]:
            print("\n➡ Backend modification flags:")
            print(result["backend_output"])

if __name__ == "__main__":
    main()
