import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as api_exceptions
from data import test_data

load_dotenv()  # Load environment variables from .env file if present

# ========== CONFIGURATION / DEBUG KEY LOADING ==========
# Try to load and normalize the API key; strip surrounding quotes if present.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # try again with stripping (handles ".env" keys wrapped in quotes)
    raw = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_API_KEY = raw.strip().strip('"').strip("'")

def _mask_key(k: str) -> str:
    if not k:
        return "None"
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}{'*'*(len(k)-8)}{k[-4:]}"

# Debug: prints presence, masked value and length (never prints full key)
print(f"DEBUG: GOOGLE_API_KEY present={bool(GOOGLE_API_KEY)} masked={_mask_key(GOOGLE_API_KEY)} length={len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0}")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing Google API key. Please set GOOGLE_API_KEY as env var or in code.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)


# ========== CORE FUNCTION ==========
def generate_team_kpi_analysis_gemini(data: dict, model_name: str = "gemini-2.5-flash") -> dict:
    """

Optimized Gemini call for KPI analysis.
- Merges observation, interpretation, and recommendation into one concise paragraph per section.
- Output must match the exact JSON format, with nested key names.
- Keeps all key insights, no loss of relevance.
- Limits each section to 2 informative lines.
- Ensures JSON-only, data-driven output. 


    """

    compact_data = json.dumps(data, separators=(",", ":"))

    prompt = (
    "You are a senior process intelligence analyst. "
    "Using the given KPI comparison data between two teams, create a concise, insight-rich benchmark analysis. "
    "Your response must be ONLY valid JSON (no markdown, no text outside JSON). "
    "Include exactly these sections: "
    "loop_analysis, bottleneck_analysis, dropout_analysis, top_5_process_variants, "
    "happy_path, recommendation_to_action, method_notes, appendix. "
    "For each section, output the field name as the key (e.g., 'loop_analysis'). "
    "Within each section, provide a single, concise paragraph of 2–4 sentences that merges the observation, interpretation, and recommendation. "
    "Focus on meaning, trends, and actionable insights. Avoid unnecessary context or filler. "
    "Ensure no important details or relationships from the KPI data are lost. "
    "Format the output exactly as shown below, with no extra explanations or markdown:"
    "\n"
    "Expected Output Format:\n"
    "{\n"
    "  \"loop_analysis\": {\n"
    "    \"loop_analysis\": \"[Compact paragraph with merged insights for loop analysis.]\"\n"
    "  },\n"
    "  \"bottleneck_analysis\": {\n"
    "    \"bottleneck_analysis\": \"[Compact paragraph with merged insights for bottleneck analysis.]\"\n"
    "  },\n"
    "  \"dropout_analysis\": {\n"
    "    \"dropout_analysis\": \"[Compact paragraph with merged insights for dropout analysis.]\"\n"
    "  },\n"
    "  \"happy_path\": {\n"
    "    \"happy_path\": \"[Compact paragraph with merged insights for happy path.]\"\n"
    "  },\n"
    "  \"recommendation_to_action\": {\n"
    "    \"recommendation_to_action\": \"[Compact paragraph with merged insights for recommendation.]\"\n"
    "  }\n"
    "  \"method_notes\": {\n"
    "    \"method_notes\": \"[Compact paragraph with merged insights for method notes.]\"\n"
    "  }\n"
    "  \"appendix\": {\n"
    "    \"appendix\": \"[Compact paragraph with merged insights for appendix.]\"\n"
    "  }\n"
    "}"
    f"KPI_DATA:{compact_data}"


)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
    except api_exceptions.NotFound as e:
        print("ERROR: The model you requested is not available for this API/version.")
        try:
            print("Fetching available models...")
            available = genai.list_models()
            print("Available models (sample):")
            # Attempt to print a concise list
            if isinstance(available, (list, tuple)):
                for m in available[:20]:
                    print("-", getattr(m, "name", m))
            elif isinstance(available, dict):
                for k in list(available.keys())[:20]:
                    print("-", k)
            else:
                print(available)
        except Exception:
            print("Could not list models programmatically. Check Google Cloud Console -> Generative AI API -> Models.")
        raise RuntimeError(
            f"Model '{model_name}' not found or unsupported for generate_content. "
            "Try 'gemini-1.5-pro' or run ListModels to see supported models."
        ) from e
    except Exception:
        raise

    # ---- Extract raw output ----
    content = ""
    if hasattr(response, "text") and response.text:
        content = response.text.strip()
    elif hasattr(response, "candidates") and response.candidates:
        try:
            content = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            pass

    # Debug: show raw output

    if not content:
        raise ValueError("Empty response from Gemini. Try using 'gemini-1.5-pro' instead of 'flash'.")

    # ---- Try parsing JSON safely ----
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end])
        else:
            raise ValueError(f"❌ Failed to parse JSON from model output:\n{content}")

    return parsed


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    result = generate_team_kpi_analysis_gemini(test_data)
    
    
    print("================ FINAL STRUCTURED REPORT ================\n")
    print(json.dumps(result, indent=4))
    print("\n=========================================================")

