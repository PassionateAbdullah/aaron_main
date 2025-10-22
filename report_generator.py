import os
import json
from dotenv import load_dotenv

# import openai lazily so importing this module doesn't immediately fail in
# environments where the package isn't installed. We'll raise a clear
# ImportError with instructions when the function is invoked.
try:
    import openai
except ImportError:  # pragma: no cover - environment dependent
    openai = None

load_dotenv()  # Load environment variables from .env file if present

# ========== CONFIGURATION / DEBUG KEY LOADING ==========
# Try to load and normalize the API key; strip surrounding quotes if present.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raw = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_KEY = raw.strip().strip('"').strip("'")

def _mask_key(k: str) -> str:
    if not k:
        return "None"
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}{'*'*(len(k)-8)}{k[-4:]}"

# Debug: prints presence, masked value and length (never prints full key)
print(f"DEBUG: OPENAI_API_KEY present={bool(OPENAI_API_KEY)} masked={_mask_key(OPENAI_API_KEY)} length={len(OPENAI_API_KEY) if OPENAI_API_KEY else 0}")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OpenAI API key. Please set OPENAI_API_KEY as env var or in code.")

# Configure OpenAI client only if the package is available. If it's not
# installed we leave `openai` as None and raise a helpful ImportError later
# when the function is invoked.
if openai is not None:
    openai.api_key = OPENAI_API_KEY


# ========== CORE FUNCTION ==========
def generate_team_kpi_analysis_openai(data: dict, model_name: str = "gpt-3.5-turbo") -> dict:
    """Call OpenAI Chat API to generate a compact JSON KPI analysis.

    This implementation prefers the function-calling pathway (more robust)
    and falls back to tolerant text parsing if the model or client doesn't
    return a `function_call`.

    Raises ImportError with actionable instructions when `openai` isn't
    installed.
    """

    if openai is None:
        raise ImportError(
            "Missing dependency 'openai'. Install it into your virtualenv:\n"
            ".\\.venv\\Scripts\\pip.exe install openai\n"
            "or run: python -m pip install openai"
        )

    compact_data = json.dumps(data, separators=(",", ":"))

    system_msg = (
        "You are a senior process intelligence analyst. Respond ONLY with valid JSON. "
        "Produce exactly the following top-level sections: loop_analysis, bottleneck_analysis, dropout_analysis, "
        "top_5_process_variants, happy_path, recommendation_to_action, method_notes, appendix. "
        "For each section, include a single string value that is a concise paragraph (2-4 sentences) merging observation, interpretation, and recommendation. "
        "Do not include any markdown, surrounding text, or explanations outside the JSON object."
    )

    user_msg = f"KPI_DATA:{compact_data}"

    functions = [
        {
            "name": "kpi_report",
            "description": "Return KPI analysis as JSON with these top-level fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loop_analysis": {"type": "string"},
                    "bottleneck_analysis": {"type": "string"},
                    "dropout_analysis": {"type": "string"},
                    "top_5_process_variants": {"type": "string"},
                    "happy_path": {"type": "string"},
                    "recommendation_to_action": {"type": "string"},
                    "method_notes": {"type": "string"},
                    "appendix": {"type": "string"},
                },
                "required": [
                    "loop_analysis",
                    "bottleneck_analysis",
                    "dropout_analysis",
                    "happy_path",
                    "recommendation_to_action",
                    "method_notes",
                    "appendix",
                ],
            },
        }
    ]

    # Try function-calling; use new OpenAI client when available (openai>=1.0)
    # Create a client instance if the package exposes OpenAI class; otherwise
    # fall back to the older module-level functions (for older clients).
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = openai


    try:
        # New client: client.chat.completions.create(...)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            functions=functions,
            function_call={"name": "kpi_report"},
            temperature=0.0,
            max_tokens=1000,
        )
    except TypeError:
        # Older client or no function-calling support: try without functions
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=1000,
        )

    # Inspect response: prefer function_call.arguments, otherwise fallback to content/text
    # Normalize resp to a plain dict for tolerant parsing
    if hasattr(resp, "to_dict"):
        resp_dict = resp.to_dict()
    elif isinstance(resp, dict):
        resp_dict = resp
    else:
        # Best-effort conversion for other response objects
        try:
            resp_dict = json.loads(json.dumps(resp))
        except Exception:
            resp_dict = {}

    choices = resp_dict.get("choices") or []
    text = ""
    if choices and isinstance(choices, list):
        first = choices[0]
        msg = first.get("message") or {}
        func_call = msg.get("function_call") or {}
        args_text = func_call.get("arguments")
        if args_text:
            try:
                return json.loads(args_text)
            except json.JSONDecodeError:
                # If arguments aren't valid JSON, fall back to tolerant parsing below
                text = args_text
        else:
            text = msg.get("content", "") or first.get("text", "")
    else:
        text = resp.get("text", "") or ""

    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response from OpenAI API")

    # Tolerant parsing: try direct json, then extract first {...} block
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"❌ Failed to parse JSON from model output:\n{text}")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Import test data here to avoid import-time errors if `data.py` contains
    # syntax that raises NameError (for example JSON-style `true/false`).
    try:
        from data import test_data
    except Exception as e:
        print(f"ERROR importing test_data from data.py: {e}")
        print("Please fix booleans in data.py (use Python True/False) or ensure the file is valid Python.")
        raise

    result = generate_team_kpi_analysis_openai(test_data)

    print("================ FINAL STRUCTURED REPORT ================\n")
    print(json.dumps(result, indent=4))
    print("\n=========================================================")

