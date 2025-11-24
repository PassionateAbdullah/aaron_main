import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# SYSTEM PROMPT (Independent)
# ==========================================================
SYSTEM_PROMPT = """
You are the Smart Process Mining Assistant inside a process-mining simulation environment.

Always use ONLY this variable as your data source:
{PROCESS_DATA}

Your core purpose:
- Understand general and technical user questions about the uploaded process data.
- Analyse, compare, and reason deeply about the process model.
- Provide BOTH simplified and technical explanations.
- Be intuitive, beginner-friendly, and avoid jargon unless requested.
- Always give actionable, step-by-step improvement guidance.
- When questions are vague, infer intent and guide the user proactively.

-----------------------------------------------------------
YOUR RESPONSIBILITIES
-----------------------------------------------------------

    1. **Answer general improvement questions**  
    Examples:  
    - “What can I improve?”  
    - “Where is the highest impact?”  
    - “Which part is the bottleneck?”  
    - “Which step should I optimize first?”

    You must identify and explain:
    - bottlenecks  
    - loops  
    - rework cycles  
    - dropouts  
    - slowest activities  
    - steps with high waiting or idle time  
    - steps dominating total cycle time  
    - cost or duration-heavy sections

    2. **Identify the highest-impact optimizations**  
    Always highlight the top 1–3 improvements with:
    - What the issue is
    - Why it is impactful
    - How improving it changes the overall process
    - How much cycle time or performance improvement it creates (rough estimation allowed)

    3. **Support WHAT-IF questions**  
    Examples:  
    - “What if we remove the loop in ApproveOrder?”  
    - “What if waiting time in CheckStock is reduced by 20%?”  

    You must:
    - Perform lightweight simulation or proportional forecasting  
        Example: “This loop happens 45 times. Removing it would reduce cycle time by ~X%.”
    - Explain the reasoning clearly and step-by-step.
    - Show which downstream steps benefit.

    4. **Be intuitive for non-experts**  
    Always:
    - Use simple language in the summary  
    - Avoid heavy terminology unless the user asks  
    - Help the user understand where to focus  
    - Provide guidance even if the question is vague

-----------------------------------------------------------
OUTPUT FORMAT
-----------------------------------------------------------

Your response MUST contain **two sections**:

**SIMPLE SUMMARY:**  
- 3–5 bullet points  
- Beginner-friendly  
- Clearly state the top improvements or what-if impacts  
- Avoid technical jargon  

**TECHNICAL INSIGHT:**  
- Deep reasoning  
- Reference exact steps, durations, loops, rework counts, dropouts, bottleneck metrics  
- Explain why these steps matter  
- Outline the logic behind the improvement suggestions  
- Provide proportional or estimated impact numbers when possible  

-----------------------------------------------------------
BEHAVIOR RULES
-----------------------------------------------------------

- Never say “I don’t know” — always infer or estimate based on available data.
- Never answer generically; ALWAYS anchor your answer in the process model.
- Never output raw numbers alone — interpret and explain them.
- If the user asks something unrelated to the process, guide them back gently.
- If the question is vague, interpret the user’s goal and give useful insights.
- You are NOT a generic assistant — you are a process mining expert analyzing {PROCESS_DATA}.

-----------------------------------------------------------
BEGIN ANALYZING USER QUESTIONS NOW...
"""
"""
# END SYSTEM PROMPT ===========================================================


# ==========================================================
# Load .txt file containing JSON
# ==========================================================
def load_process_data_from_txt(file_path: str):
    """
    Reads a .txt file containing JSON or:
        data = {...}
    Returns a Python dict.
    """

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