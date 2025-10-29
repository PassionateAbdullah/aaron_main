import re
import os
import json
from typing import Dict, Optional

from dotenv import load_dotenv
try:
    # Align with existing project usage (see Landing_page_bot.py)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for simpler matching."""
    t = text.lower()
    # Normalize common punctuation to spaces
    t = re.sub(r"[\t\n\r]+", " ", t)
    t = re.sub(r"[\-_/\\,:;.!?()\[\]{}]", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _contains_pair(text: str, verbs, nouns) -> bool:
    """Return True if any verb appears near any noun (order-agnostic)."""
    v = r"(?:" + "|".join(map(re.escape, verbs)) + r")"
    n = r"(?:" + "|".join(map(re.escape, nouns)) + r")"
    # verb ... noun OR noun ... verb, within a short window
    pattern = rf"\b{v}\b\W{{0,40}}\b{n}\b|\b{n}\b\W{{0,40}}\b{v}\b"
    return re.search(pattern, text) is not None


def _extract_reduce_percentage(text: str) -> Optional[float]:
    """
    Extract a percentage the user specified for reduction of cost/time.
    Returns the numeric percentage value if present, else None.
    Examples matched:
      - reduce cost 5%
      - lower time by 12.5 %
      - cut cose to 0.2%
    """
    # Allow a common misspelling "cose" for cost
    target = r"(?:cost|time|cose)"
    verbs = r"(?:reduce|decrease|lower|cut|drop|lessen|minimi[sz]e)"
    m = re.search(rf"\b{verbs}\b\W*(?:the\s*)?{target}\b\W*(?:by|to)?\W*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_user_intent(user_input: str) -> Dict[str, object]:
    """
    Parse free-form user input and infer intents about showing/removing
    loops, bottlenecks, dropout, and reducing cost/time.

    Returns a dict with normalized boolean flags and optional percentage:
      {
        'show_loops': bool,
        'show_bottlenecks': bool,
        'show_dropout': bool,
        'remove_loops': bool,
        'remove_bottlenecks': bool,
        'remove_dropout': bool,
        'reduce_cost': bool,                 # cost or time reduction intent
        'reduce_cost_by_percent': float|None # percentage if specified
      }

    Rules:
    - "show" intents: show/display/visualize/highlight + loops/bottlenecks/dropout
    - "remove" intents: remove/eliminate/fix/cut/drop/reduce + loops/bottlenecks/dropout
    - "reduce cost" also triggers removal intents for loops, bottlenecks, and dropout
      because cost is tied to these inefficiencies.
    - "reduce time" is treated the same as reducing cost.
    """
    text = _normalize_text(user_input)

    # Vocabulary
    show_verbs = [
        "show", "display", "visualize", "highlight", "list", "reveal", "see", "plot"
    ]
    remove_verbs = [
        "remove", "eliminate", "delete", "drop", "fix", "resolve", "cut", "reduce"
    ]

    loop_nouns = ["loop", "loops", "looping", "rework", "reworks"]
    bottleneck_nouns = ["bottleneck", "bottlenecks", "bottlenecking", "constraint", "constraints"]
    dropout_nouns = ["dropout", "dropouts", "dropped", "abandonment", "abandonments"]

    # Show intents
    show_loops = _contains_pair(text, show_verbs, loop_nouns)
    show_bottlenecks = _contains_pair(text, show_verbs, bottleneck_nouns)
    show_dropout = _contains_pair(text, show_verbs, dropout_nouns)

    # Remove intents
    remove_loops = _contains_pair(text, remove_verbs, loop_nouns)
    remove_bottlenecks = _contains_pair(text, remove_verbs, bottleneck_nouns)
    remove_dropout = _contains_pair(text, remove_verbs, dropout_nouns)

    # Cost/time reduction intent (includes common misspelling "cose")
    cost_time_verbs = r"reduce|decrease|lower|cut|drop|lessen|minimi[sz]e|optimis|optimiz|save"
    if re.search(rf"\b(?:{cost_time_verbs})\b\W*(?:the\s*)?(?:cost|time|cose)\b", text):
        reduce_cost = True
    else:
        # Phrases like "cost reduction" or "time reduction"
        reduce_cost = re.search(r"\b(cost|time)\b\W*(reduction|down)\b", text) is not None

    reduce_cost_by_percent = _extract_reduce_percentage(text)

    # If user intends to reduce cost/time, we also want to remove inefficiencies
    if reduce_cost:
        remove_loops = True or remove_loops
        remove_bottlenecks = True or remove_bottlenecks
        remove_dropout = True or remove_dropout

    return {
        "show_loops": bool(show_loops),
        "show_bottlenecks": bool(show_bottlenecks),
        "show_dropout": bool(show_dropout),
        "remove_loops": bool(remove_loops),
        "remove_bottlenecks": bool(remove_bottlenecks),
        "remove_dropout": bool(remove_dropout),
        "reduce_cost": bool(reduce_cost),
        "reduce_cost_by_percent": reduce_cost_by_percent,
    }


# ===================== LLM SYSTEM PROMPT =====================
SYSTEM_PROMPT_LOW_CODE = """
You are an intent-extraction assistant for process analytics.
Goal: Convert user text into a strict JSON object describing what to show or remove (loops, bottlenecks, dropout) and whether to reduce cost/time.

Important business rule:
- If the user asks to reduce cost OR reduce time, then also set remove_loops, remove_bottlenecks, and remove_dropout to true because these inefficiencies drive cost/time.

Output format (JSON only, no prose):
{
  "show_loops": boolean,
  "show_bottlenecks": boolean,
  "show_dropout": boolean,
  "remove_loops": boolean,
  "remove_bottlenecks": boolean,
  "remove_dropout": boolean,
  "reduce_cost": boolean,                  // true if user wants to reduce cost or time
  "reduce_cost_by_percent": number|null    // percentage value if explicitly given, else null
}

Guidelines:
- Accept synonyms: show/display/visualize/highlight for show; remove/eliminate/fix/cut/reduce for remove; cost may be misspelled as "cose"; "reduce time" is treated the same as "reduce cost".
- If a percentage is specified for cost/time reduction (e.g., 0.2%, 5%), set reduce_cost to true and provide the numeric value in reduce_cost_by_percent.
- If something is not mentioned, set its value to false (or null for the percentage field).
- Respond with valid JSON only. Do not include comments or extra text.
"""


_openai_client = None


def _get_openai_client():  # lazy init
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing; cannot call OpenAI API.")
    if OpenAI is None:
        raise ImportError("openai SDK not installed. Run: pip install openai python-dotenv")
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def llm_parse_user_intent(user_input: str, model: str = "gpt-4o-mini") -> Dict[str, object]:
    """
    Low-code LLM wrapper: sends the system prompt + user input to OpenAI and
    expects a strict JSON object. Falls back to local parser if JSON parsing fails.
    """
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_LOW_CODE},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)

        # Ensure keys exist and normalize types
        out = {
            "show_loops": bool(data.get("show_loops", False)),
            "show_bottlenecks": bool(data.get("show_bottlenecks", False)),
            "show_dropout": bool(data.get("show_dropout", False)),
            "remove_loops": bool(data.get("remove_loops", False)),
            "remove_bottlenecks": bool(data.get("remove_bottlenecks", False)),
            "remove_dropout": bool(data.get("remove_dropout", False)),
            "reduce_cost": bool(data.get("reduce_cost", False)),
            "reduce_cost_by_percent": (
                float(data["reduce_cost_by_percent"]) if data.get("reduce_cost_by_percent") is not None else None
            ),
        }

        # Enforce the business rule server-side in case the model missed it
        if out["reduce_cost"]:
            out["remove_loops"] = True
            out["remove_bottlenecks"] = True
            out["remove_dropout"] = True
        return out
    except Exception:
        # Fallback to deterministic local parsing
        return parse_user_intent(user_input)


if __name__ == "__main__":
    # Simple manual tests
    samples = [
        "show loop and bottlenecks",
        "remove bottleneck, reduce cose 0.2%",
        "please reduce cost by 12.5% and remove dropouts",
        "can you highlight dropout but don’t change anything",
        "lower time",
    ]
    print("\nLocal parser results:\n")
    for s in samples:
        print(s, "->", parse_user_intent(s))

    # If OPENAI_API_KEY available, also show LLM results for comparison
    if os.getenv("OPENAI_API_KEY"):
        print("\nLLM parser results (low-code):\n")
        for s in samples:
            print(s, "->", llm_parse_user_intent(s))
