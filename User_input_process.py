import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """You are a process analytics assistant. Parse requests into JSON following these rules:

1. "show/analyze" = Analysis Mode: set case field, all remove_* = false
2. "remove/reduce" = Action Mode: case = null, set relevant remove_* true
3. Detect process name for target_activity
4. Capture any percentage mentioned (20% → 20)
5. "reduce cost/time" sets all remove_* true

Output format:
{
    "remove_bottlenecks": boolean,
    "remove_loops": boolean,
    "remove_dropouts": boolean,
    "target_activity": string | null,
    "case": string | null,
    "target_percentage": number | null
}

Key examples:
"show loops in Invoice" →
{
    "remove_bottlenecks": false,
    "remove_loops": false,
    "remove_dropouts": false,
    "target_activity": "Invoice",
    "case": "Loop",
    "target_percentage": null
}

"remove 20% bottlenecks from Payment" →
{
    "remove_bottlenecks": true,
    "remove_loops": false,
    "remove_dropouts": false,
    "target_activity": "Payment",
    "case": null,
    "target_percentage": 20
}

Return ONLY the JSON with no additional text or explanation."""

def parse_process_intent(user_input: str) -> Dict[str, object]:
    # Load OpenAI API key from environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Get LLM's interpretation of the user's intent
        response = client.chat.completions.create(
            model="gpt-4",  # Can be changed to gpt-3.5-turbo for lower latency
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,  # Use 0 for consistent, deterministic outputs
            max_tokens=150
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content.strip())
        
        # Ensure we have the exact structure we want
        return {
            "remove_bottlenecks": bool(result.get("remove_bottlenecks", False)),
            "remove_loops": bool(result.get("remove_loops", False)),
            "remove_dropouts": bool(result.get("remove_dropouts", False)),
            "target_activity": str(result["target_activity"]) if result.get("target_activity") else None,
            "case": str(result["case"]) if result.get("case") else None,
            "target_percentage": float(result["target_percentage"]) if result.get("target_percentage") else None
        }
    except Exception as e:
        # If anything fails, return a safe default
        print(f"Error processing input: {str(e)}")
        return {
            "remove_bottlenecks": False,
            "remove_loops": False,
            "remove_dropouts": False,
            "target_activity": None,
            "case": None,
            "target_percentage": None,
        }


if __name__ == "__main__":
    print("Process Analytics Assistant")
    print("Type 'exit' or press Ctrl+C to quit\n")
    
    while True:
        try:
            user_text = input("Your request: ").strip()
            if not user_text or user_text.lower() == 'exit':
                break
                
            result = parse_process_intent(user_text)
            print("\nOutput:")
            print(json.dumps(result, indent=4))
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            
    print("\nGoodbye!")
