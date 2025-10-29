import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """You are a process analytics assistant that converts user requests into structured actions.

Rules for understanding user intent:
1. If user mentions "show" or similar words (display/visualize/see), they want to analyze that aspect
2. If user mentions "reduce cost" or "reduce time", set all removal flags to true as this requires removing all inefficiencies
3. "Remove" or similar words (eliminate/fix/delete) for a specific item sets only that flag true
4. Target activity is the specific process or area mentioned (e.g., "in Payment Processing", "during Customer Service")
5. Watch for implicit meanings:
   - "optimize Payment Processing" implies removing bottlenecks
   - "improve efficiency in Sales" implies removing bottlenecks
   - "fix the process" implies removing all issues

Return ONLY valid JSON matching this exact structure:
{
    "remove_bottlenecks": boolean,
    "remove_loops": boolean,
    "remove_dropouts": boolean,
    "target_activity": string | null
}

Example inputs and outputs:
Input: "show me the loops in Payment Processing"
{
    "remove_bottlenecks": false,
    "remove_loops": true,
    "remove_dropouts": false,
    "target_activity": "Payment Processing"
}

Input: "reduce processing time"
{
    "remove_bottlenecks": true,
    "remove_loops": true,
    "remove_dropouts": true,
    "target_activity": null
}

Input: "remove bottlenecks from Order Processing"
{
    "remove_bottlenecks": true,
    "remove_loops": false,
    "remove_dropouts": false,
    "target_activity": "Order Processing"
}

Return ONLY the JSON with no additional text or explanation."""

def parse_process_intent(user_input: str) -> Dict[str, object]:
    """
    Use OpenAI to understand user intent and return a structured response:
    {
        "remove_bottlenecks": bool,
        "remove_loops": bool,
        "remove_dropouts": bool,
        "target_activity": str | None
    }
    """
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
            "target_activity": str(result["target_activity"]) if result.get("target_activity") else None
        }
    except Exception as e:
        # If anything fails, return a safe default
        print(f"Error processing input: {str(e)}")
        return {
            "remove_bottlenecks": False,
            "remove_loops": False,
            "remove_dropouts": False,
            "target_activity": None
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
