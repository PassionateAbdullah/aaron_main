import os
import json
import re
from typing import Dict, Optional
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SYSTEM_PROMPT = """
You are a process analytics assistant. Convert user requests about process improvements into a simple action format.

Return only valid JSON matching this structure:
{
    "remove_bottlenecks": boolean,
    "remove_loops": boolean,
    "remove_dropouts": boolean,
    "target_activity": string | null
}

Use exactly these field names.
"""


def _fallback_parse(user_input: str) -> Dict[str, object]:
    text = user_input.lower()
    remove_bottlenecks = bool(re.search(r"\b(bottleneck|bottlenecks|constraint|constraints)\b", text))
    remove_loops = bool(re.search(r"\b(loop|loops|rework|reworks)\b", text))
    remove_dropouts = bool(re.search(r"\b(dropout|dropouts|abandonment|abandonments|dropped)\b", text))

    # Try to extract activity after 'in', 'from', or 'for'
    m = re.search(r"\b(?:in|from|for)\s+([^,;\n]+)", user_input, flags=re.I)
    target_activity = None
    if m:
        act = m.group(1).strip()
        # Stop at common conjunctions
        act = re.split(r"\band\b|\bfor\b|\bto\b|\bwith\b", act, flags=re.I)[0].strip()
        # Normalize whitespace and strip trailing punctuation
        target_activity = act.strip(" .;,") or None

    return {
        "remove_bottlenecks": remove_bottlenecks,
        "remove_loops": remove_loops,
        "remove_dropouts": remove_dropouts,
        "target_activity": target_activity,
    }


def parse_process_intent(user_input: str, model: str = "gpt-4") -> Dict[str, object]:
    """Return structure:
    {
      "remove_bottlenecks": bool,
      "remove_loops": bool,
      "remove_dropouts": bool,
      "target_activity": str|None
    }
    Uses the OpenAI client if available; otherwise falls back to a simple regex-based parser.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            return {
                "remove_bottlenecks": bool(data.get("remove_bottlenecks", False)),
                "remove_loops": bool(data.get("remove_loops", False)),
                "remove_dropouts": bool(data.get("remove_dropouts", False)),
                "target_activity": str(data.get("target_activity")) if data.get("target_activity") else None,
            }
        except Exception:
            # Fall through to local parser
            return _fallback_parse(user_input)
    else:
        return _fallback_parse(user_input)


if __name__ == "__main__":
    load_dotenv()
    tests = [
        "remove bottlenecks in Payment Monitoring",
        "eliminate loops and dropouts from Customer Onboarding",
        "fix bottlenecks and dropouts in Order Processing",
        "please remove dropouts for Payment Monitoring and optimize"
    ]

    for t in tests:
        out = parse_process_intent(t)
        print("\nInput:", t)
        print(json.dumps(out, indent=4))