import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
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

1. General / Non-technical Question
   - Examples: "What's wrong?", "What can I improve?", "Where is bottleneck?"
   - Respond with:
       SIMPLE EXPLANATION ONLY
       - short
       - beginner-friendly
       - plain language, no jargon
       - main insights in 2–4 bullets

2. Technical / Expert Question
   - Examples: "Cycle time root cause?", "Variant distribution?", 
               "Impact of loop removal?", "Throughput bottleneck analysis"
   - Respond with:
       TECHNICAL ANALYSIS
       - bottlenecks, loops, rework, dropouts
       - durations, frequencies, counts
       - step-by-step reasoning
       - proportional or estimated impacts

3. If mixed or unclear
   - Give a short simple summary first
   - THEN provide a concise technical explanation

------------------------------------------------
RULES
------------------------------------------------
- Use ONLY the dataset provided.
- Never make up data.
- Never answer with generic text.
- If user asks off-topic, gently guide back.
- If user requests deeper or simpler explanation, adapt instantly.
"""


def load_process_data_from_json(file_path: str):
    """Load process data from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Process data file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_query(msg: str) -> str:
    """Normalize user queries for better LLM performance."""
    msg = msg.strip().lower()
    replacements = {
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


def generate_process_mining_response(user_message: str, process_data: dict) -> str:
    """
    Sends system prompt + process data + normalized user query to OpenAI API.
    Returns ONLY the model's text response.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Fill system prompt with process data
    system_prompt_filled = SYSTEM_PROMPT.replace(
        "{PROCESS_DATA}", json.dumps(process_data, indent=2)
    )
    
    # Normalize user message
    normalized_user_msg = normalize_query(user_message)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_filled},
                {"role": "user", "content": normalized_user_msg}
            ]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"[ERROR] AI request failed: {str(e)}"


# Test code (only runs when script is executed directly)
if __name__ == "__main__":
    try:
        process_data = load_process_data_from_json("aaron_data.json")
        question = "where is problem"
        answer = generate_process_mining_response(question, process_data)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")