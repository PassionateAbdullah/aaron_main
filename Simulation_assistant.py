# ==========================================================
# SMART PROCESS MINING CHATBOT — FINAL PATCHED VERSION
# ==========================================================

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==========================================================
# 0. CONVERSATION MEMORY (last 2 analysis responses only)
# ==========================================================

CONVERSATION_MEMORY = {
    "analysis_history": []
}

def save_analysis_to_memory(text: str):
    history = CONVERSATION_MEMORY["analysis_history"]
    history.append(text)
    if len(history) > 3:     # store only last 2 analyses
        history.pop(0)



# ==========================================================
# 1. TEXT NORMALIZER
# ==========================================================

NORMALIZER_PROMPT = """
You are a text normalizer.

Rewrite the user’s message into a clear and explicit instruction without changing meaning.
Return ONLY the rewritten message.
"""

def normalize_query(msg: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": NORMALIZER_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()

    except:
        return msg.strip()



# ==========================================================
# 2. BASE INTENT CLASSIFIER (greeting / action / analysis)
# ==========================================================

INTENT_CLASSIFIER_PROMPT = """
Classify the user's message into ONLY one:

1. "greeting"
2. "action"
3. "analysis"

Rules:
- "action": user wants to change/remove loops, bottlenecks, dropouts.
- "analysis": user asks questions about insights, metrics, bottlenecks, comparisons.
- "greeting": hello/hi/etc.

Return ONLY the label.
"""

def detect_intent(msg: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0,
            max_tokens=10
        )
        label = res.choices[0].message.content.strip().lower()
        return label if label in ["greeting", "action", "analysis"] else "analysis"

    except:
        return "analysis"



# ==========================================================
# 3. FOLLOW-UP INTENT CLASSIFIER (fixed)
# ==========================================================

FOLLOWUP_PROMPT = """
Classify this into EXACTLY one:

1. "followup_summary"
   - user wants summary / bullet points of previous analysis
   - MUST refer to previous context (explicit or implicit)
   - examples: summarize it, short version, bullet points

2. "followup_analysis"
   - user wants deeper explanation of previous analysis
   - MUST refer to previous context
   - examples: explain that more, expand previous part, analyze that again

3. "followup_action"
   - user wants to MODIFY the model
   - keywords: fix, clean, remove, reduce, eliminate loops/dropouts/bottlenecks
   - short commands like "clean loops", "reduce dropouts only" MUST be action

4. "new_question"
   - new analysis request
   - examples: analyze ShipGoods, bottleneck analysis, loop impact

Return ONLY one label.
"""

def detect_followup_intent(msg: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FOLLOWUP_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0
        )
        return res.choices[0].message.content.strip().lower()

    except:
        return "new_question"



# ==========================================================
# 4. FOLLOW-UP HANDLERS
# ==========================================================

def handle_followup_summary(user_msg: str):
    """Summarize last analysis only."""
    if not CONVERSATION_MEMORY["analysis_history"]:
        return "There is no previous analysis to summarize."

    last = CONVERSATION_MEMORY["analysis_history"][-1]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PROMPT = f"""
Summarize the following analysis in clear bullet points:

{last}

User request: "{user_msg}"
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": PROMPT}],
        temperature=0.1,
        max_tokens=180
    )
    return res.choices[0].message.content.strip()



def handle_followup_analysis(user_msg: str):
    """Deepen a previous analysis."""
    if not CONVERSATION_MEMORY["analysis_history"]:
        return "There is no previous analysis to expand."

    last = CONVERSATION_MEMORY["analysis_history"][-1]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PROMPT = f"""
Extract ONLY the parts relevant to the user follow-up request.
Do not hallucinate or introduce new information.

Previous analysis:
{last}

User follow-up: "{user_msg}"
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": PROMPT}],
        temperature=0.1,
        max_tokens=250
    )
    return res.choices[0].message.content.strip()



# ==========================================================
# 5. PROCESS MINING ENGINE
# ==========================================================

def generate_process_mining_response(msg: str, dataset: dict) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    SYSTEM_PROMPT = f"""
You are a mathematical process-mining engine.

Use ONLY the dataset below:
{json.dumps(dataset, indent=2)}

Rules:
- Use real numbers only, no hallucinations.
- Max 6–8 bullet points.
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return res.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] {e}"



# ==========================================================
# 6. ACTION PARSER (JSON mode)
# ==========================================================

ACTION_PROMPT = """
Convert the user's request into strict JSON:

{
  "remove_bottlenecks": boolean,
  "remove_loops": boolean,
  "remove_dropouts": boolean
}

Return ONLY this JSON.
"""

def parse_process_intent(msg: str) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ACTION_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0
        )
        parsed = json.loads(res.choices[0].message.content.strip())
        return {
            "remove_bottlenecks": bool(parsed.get("remove_bottlenecks", False)),
            "remove_loops": bool(parsed.get("remove_loops", False)),
            "remove_dropouts": bool(parsed.get("remove_dropouts", False)),
        }

    except:
        return {
            "remove_bottlenecks": False,
            "remove_loops": False,
            "remove_dropouts": False,
        }



# ==========================================================
# 7. MAIN ROUTER (patched logic)
# ==========================================================

def dynamic_process_chatbot(msg: str, dataset: Dict[str, Any]):
    normalized = normalize_query(msg)

    # FIRST: if no previous analysis → it CANNOT be follow-up.
    if len(CONVERSATION_MEMORY["analysis_history"]) == 0:
        follow = "new_question"
    else:
        follow = detect_followup_intent(normalized)

    # FOLLOW-UP ACTION
    if follow == "followup_action":
        return parse_process_intent(normalized)

    # FOLLOW-UP SUMMARY
    if follow == "followup_summary":
        return handle_followup_summary(normalized)

    # FOLLOW-UP ANALYSIS
    if follow == "followup_analysis":
        return handle_followup_analysis(normalized)

    # Otherwise → classic intent classifier
    intent = detect_intent(normalized)

    if intent == "greeting":
        return "Hey! 😊 I'm your Simulation Assistant. How can I help you today?"

    if intent == "analysis":
        result = generate_process_mining_response(normalized, dataset)
        save_analysis_to_memory(result)
        return result

    # ACTION MODE
    return parse_process_intent(normalized)



# ==========================================================
# 8. LOAD DATA FILE
# ==========================================================

def load_process_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Try multiple encodings to handle BOM and other issues
    with open(path, "rb") as f:
        raw = f.read()
    
    # Try to decode with different encodings
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = raw.decode(encoding)
            return json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    
    # If all fail, raise error
    raise ValueError(f"Could not decode {path} with any encoding (UTF-8-sig, UTF-8, Latin-1)")


# ==========================================================
# BACKEND WRAPPER — PRODUCTION READY
# ==========================================================

def process_query(user_message: str, process_data: dict) -> dict:
    """
    Main backend wrapper.
    
    Input:
        user_message (str)  - user input from frontend/backend
        process_data (dict) - dataset loaded from aaron_data.json
    
    Output (dict):
        {
            "response_type": "analysis" | "action" | "summary" | "followup" | "error",
            "content": "text response",
            "action_flags": {...}        # only when action
        }
    """

    try:
        normalized = normalize_query(user_message)

        # Determine follow-up mode (only if memory exists)
        if len(CONVERSATION_MEMORY["analysis_history"]) == 0:
            follow = "new_question"
        else:
            follow = detect_followup_intent(normalized)

        # -----------------------------
        # FOLLOW-UP ACTION
        # -----------------------------
        if follow == "followup_action":
            flags = parse_process_intent(normalized)
            return {
                "response_type": "action",
                "content": "",
                "action_flags": flags
            }

        # -----------------------------
        # FOLLOW-UP SUMMARY
        # -----------------------------
        if follow == "followup_summary":
            summary = handle_followup_summary(normalized)
            return {
                "response_type": "summary",
                "content": summary,
                "action_flags": None
            }

        # -----------------------------
        # FOLLOW-UP ANALYSIS
        # -----------------------------
        if follow == "followup_analysis":
            deeper = handle_followup_analysis(normalized)
            return {
                "response_type": "followup",
                "content": deeper,
                "action_flags": None
            }

        # -----------------------------
        # NEW QUESTION → INTENT CLASSIFIER
        # -----------------------------
        intent = detect_intent(normalized)

        # GREETING
        if intent == "greeting":
            return {
                "response_type": "greeting",
                "content": "Hey! 😊 I'm your Simulation Assistant. How can I help you today?",
                "action_flags": None
            }

        # ANALYSIS
        if intent == "analysis":
            analysis = generate_process_mining_response(normalized, process_data)
            save_analysis_to_memory(analysis)
            return {
                "response_type": "analysis",
                "content": analysis,
                "action_flags": None
            }

        # ACTION MODE
        flags = parse_process_intent(normalized)
        return {
            "response_type": "action",
            "content": "",
            "action_flags": flags
        }

    except Exception as e:
        return {
            "response_type": "error",
            "content": f"Backend error: {str(e)}",
            "action_flags": None
        }


# # ==========================================================
# # 9. CLI RUNNER
# # ==========================================================

# def run_chatbot():
#     print("🟢 Smart Process Mining Chatbot (Final Patched Version)")
#     print("Type 'exit' to quit.\n")

#     # Reset memory at the start of each session
#     CONVERSATION_MEMORY["analysis_history"] = []

#     try:
#         dataset = load_process_data("aaron_data.json")
#     except Exception as e:
#         print("[Error loading dataset]:", e)
#         return

#     while True:
#         user = input("You: ").strip()

#         if user.lower() in ["exit", "quit"]:
#             print("Goodbye! 👋")
#             break

#         response = dynamic_process_chatbot(user, dataset)

#         print("\n--- Chatbot Response ---")
#         print(response)
#         print("------------------------\n")



# # ==========================================================
# # MAIN
# # ==========================================================

# if __name__ == "__main__":
#     run_chatbot()

