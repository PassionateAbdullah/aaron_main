import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# SYSTEM PROMPT (Independent)
# ==========================================================
SYSTEM_PROMPT = """
You are the Smart Process Mining Assistant.

Your ONLY data source:
{PROCESS_DATA}

Your role:
- Understand user questions about the process.
- Detect whether the user wants a simple/general or technical/deep answer.
- Automatically adapt your explanation style based on the user’s intent.
- When uncertain, start simple but offer deeper technical detail on request.

------------------------------------------------
INTENT LOGIC (Important)
------------------------------------------------
Identify user intent:

1. **General / Non-technical Question**
   - Examples: “What’s wrong?”, “What can I improve?”, “Where is bottleneck?”
   - Respond with:
       SIMPLE EXPLANATION ONLY
       - short
       - beginner-friendly
       - plain language, no jargon
       - main insights in 2–4 bullets

2. **Technical / Expert Question**
   - Examples: “Cycle time root cause?”, “Variant distribution?”, 
               “Impact of loop removal?”, “Throughput bottleneck analysis”
   - Respond with:
       TECHNICAL ANALYSIS
       - bottlenecks, loops, rework, dropouts
       - durations, frequencies, counts
       - step-by-step reasoning
       - proportional or estimated impacts

3. **If mixed or unclear**
   - Give a short simple summary first
   - THEN provide a concise technical explanation

------------------------------------------------
WHAT YOU MUST ANALYZE (When Needed)
------------------------------------------------
- bottlenecks
- loops & rework cycles
- dropouts
- slowest activities
- variant patterns
- cost or duration-heavy steps
- what-if improvements (simulate proportionally)

------------------------------------------------
RULES
------------------------------------------------
- Use ONLY the dataset provided.
- Never make up data.
- Never answer with generic text.
- If user asks off-topic, gently guide back.
- If user requests deeper or simpler explanation, adapt instantly.

Begin analyzing user questions now.
"""
# END SYSTEM PROMPT ===========================================================


# ==========================================================
# Load .txt file containing JSON
# ==========================================================
def load_process_data_from_txt(file_path: str):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        raw = f.read().strip()

    # If formatted as: data = {...}
    if raw.startswith("data"):
        _, json_part = raw.split("=", 1)
        raw = json_part.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON inside TXT file: {e}")


# ==========================================================
# Normalize user queries for better LLM performance
# ==========================================================
def normalize_query(msg: str) -> str:
    msg = msg.strip().lower()

    replacements = {
        "bottleneck": "where is the bottleneck in the process?",
        "improve": "what can be improved in the process?",
        "help": "explain the process issues and improvements",
        "slow": "which activities are slow and causing delays?",
        "loop": "which steps have loops or rework cycles?",
        "dropout": "which steps have dropouts?",
        "rework": "which steps have rework cycles?",
        "bottleneck": "where is the bottleneck in the process?",
        "improve": "what can be improved in the process?",
        "help": "explain the process issues and improvements",
        "slow": "which activities are slow and causing delays?",
        "loop": "which steps have loops or rework cycles?",
        "dropout": "which steps have dropouts?",
        "rework": "which steps have rework cycles?",
        "bottleneck": "where is the bottleneck in the process?",
        "improve": "what can be improved in the process?",
        "help": "explain the process issues and improvements",
        "slow": "which activities are slow and causing delays?",
        "loop": "which steps have loops or rework cycles?",
        "dropout": "which steps have dropouts?",
        "rework": "which steps have rework cycles?",

    }

    for key, val in replacements.items():
        if key in msg:
            return val

    return msg


# ==========================================================
# Generate Process Mining AI Response
# ==========================================================
def generate_process_mining_response(user_message: str, model, process_data):
    """
    Sends system prompt + process data + normalized user query to OpenAI API.
    """

    # Fill SYSTEM PROMPT with injected dataset
    system_prompt_filled = SYSTEM_PROMPT.replace(
        "{PROCESS_DATA}", json.dumps(process_data, indent=2)
    )

    user_message_norm = normalize_query(user_message)

    try:
        response = model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_filled},
                {"role": "user", "content": user_message_norm}
            ]
        )

        return {
            "status": "ok",
            "response": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Model error occurred."
        }


# ==========================================================
# MAIN CHATBOT LOOP
# ==========================================================
if __name__ == "__main__":
    print("🔵 Smart Process Mining Chatbot Started")
    print("Type your question below. Type 'exit' to quit.")
    print("---------------------------------------------------")

    # LOAD DATA FROM TXT FILE
    PROCESS_DATA = load_process_data_from_txt("aaron_data.txt")
    print("✔ Loaded process data from aaron_data.txt")

    # INIT OPENAI CLIENT
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while True:
        user_input = input("\nYou: ")

        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break

        response = generate_process_mining_response(user_input, client, PROCESS_DATA)

        if response["status"] == "ok":
            print("\nAI: " + response["response"])
        else:
            print("\nError: " + response["error"])