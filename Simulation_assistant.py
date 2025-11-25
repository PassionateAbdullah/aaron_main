from typing import Dict, Any
from process_mining_chatbot import generate_process_mining_response
from User_input_process import parse_process_intent


# Keywords
ACTION_KEYWORDS = ["remove", "reduce", "fix", "eliminate", "optimize", "clean"]
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]


# 🔥 Rebuilt system prompt using correct string formatting
SYSTEM_PROMPT_CHATBOT = """
You are the Smart Process Mining Chatbot.

Your goals:
1. Greet the user politely and naturally.
2. Understand user intent:
   - Start the conversation with any greeting in {greeting_keywords}.
   - If the user asks about the process → perform ANALYSIS.
   - If the user requests removal/reduction → perform ACTION.
3. Never show backend JSON to the user.
4. Keep user responses clear, helpful, friendly.
5. Always follow the format used by the dynamic_process_chatbot function.
""".format(
    greeting_keywords=GREETING_KEYWORDS
)


def detect_intent(user_message: str) -> str:
    """
    Detect whether the user wants:
    - 'greeting'
    - 'analysis'
    - 'action'
    """
    text = user_message.lower().strip()

    # ➤ Greeting detection (strong match)
    if any(text.startswith(g) for g in GREETING_KEYWORDS):
        return "greeting"

    # ➤ Action request detection
    if any(k in text for k in ACTION_KEYWORDS):
        return "action"

    # ➤ Default → analysis
    return "analysis"


def dynamic_process_chatbot(user_message: str, process_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified chatbot with system prompt behavior.
    Safely handles: greeting → analysis → action.
    """

    intent = detect_intent(user_message)

    # ============================
    # 🟢 GREETING MODE
    # ============================
    if intent == "greeting":
        return {
            "mode": "greeting",
            "response": "Hello! 😊 How can I help you with your process simulation today?",
            
        }

    # ============================
    # 🔵 ANALYSIS MODE
    # ============================
    if intent == "analysis":
        try:
            answer = generate_process_mining_response(
                user_message=user_message,
                process_data=process_json
            )
        except Exception as e:
            answer = f"Sorry, I couldn’t analyze the process due to an internal error: {e}"

        return {
            "mode": "analysis",
            "user_response": answer,
            "backend_output": None
        }

    # ============================
    # 🟠 ACTION MODE
    # ============================
    if intent == "action":
        try:
            parsed_json = parse_process_intent(user_message)
        except Exception:
            parsed_json = {
                "remove_bottlenecks": False,
                "remove_loops": False,
                "remove_dropouts": False
            }

        formal_message = (
            "Got it! 👍 Your optimization request is understood. "
            "We are now applying the requested improvements. "
            "The process model will be updated shortly."
        )

        return {
            "mode": "action",
            "user_response": formal_message,
            "backend_output": parsed_json
        }
