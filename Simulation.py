import json
import re
import os
import datetime
from copy import deepcopy

# Optional OpenAI integration (used to refine/confirm parsed intent)
try:
    import openai
except Exception:
    openai = None

# Import analyzer helpers (used as fallback if precomputed JSON isn't present)
try:
    from Process_mining import load_event_log_from_csv, analyze_and_structure_process_data
except Exception:
    # If import fails, we'll only support reading precomputed JSON file below.
    load_event_log_from_csv = None
    analyze_and_structure_process_data = None


# ---------------------------- 1. PARSER ----------------------------
def parse_user_input(user_input: str) -> dict:
    """
    Parse user natural language query into structured intent.
    Example:
      "remove all loops"
      "show bottlenecks"
      "reduce cost"
      "remove bottleneck 0.2%"
    """
    user_input = user_input.lower()
    intent = {
        "action": None,
        "target": None,
        "percentage": None,
        "step": None
    }

    # detect action
    if "show" in user_input or "see" in user_input or "display" in user_input:
        intent["action"] = "show"
    elif "remove" in user_input or "delete" in user_input:
        intent["action"] = "remove"
    elif "reduce" in user_input or "decrease" in user_input:
        intent["action"] = "reduce"
    elif "increase" in user_input:
        intent["action"] = "increase"

    # detect target
    for keyword in ["loop", "loops", "bottleneck", "bottlenecks", "dropout", "dropouts", "cost"]:
        if keyword in user_input:
            intent["target"] = keyword.rstrip("s")

    # extract percentage if present
    percent_match = re.search(r"(\\d+(\\.\\d+)?)\\s*%", user_input)
    if percent_match:
        intent["percentage"] = float(percent_match.group(1)) / 100.0

    # detect step (if specific activity mentioned)
    activity_match = re.search(r"(invoice\\s+created|invoice\\s+sent|payment\\s+monitoring|payment\\s+received|receipt\\s+reconciled|archive)", user_input)
    if activity_match:
        intent["step"] = activity_match.group(1).title()

    return intent


# ---------------------------- 2. DISPLAY ----------------------------
def get_requested_info(data: dict, target: str) -> dict:
    """Return only the requested info subset (loops, bottlenecks, dropouts)."""
    nodes = data.get("process_flow_nodes", [])
    filtered = []

    if target == "loop":
        filtered = [n for n in nodes if n.get("hasLoop")]
    elif target == "bottleneck":
        filtered = [n for n in nodes if n.get("isBottleneck")]
    elif target == "dropout":
        filtered = [n for n in nodes if n.get("isDropout")]

    return {"requested_view": target, "results": filtered}


# ---------------------------- 3. REMOVE LOOP ----------------------------
def remove_loops(data: dict, step_name: str = None) -> dict:
    """Remove loops globally or from specific step."""
    updated = deepcopy(data)
    for node in updated.get("process_flow_nodes", []):
        if step_name and node["label"].lower() != step_name.lower():
            continue
        if node.get("hasLoop"):
            node["hasLoop"] = False
            node["loopConnections"] = None
            node["descriptions"].append("Loop removed from this step.")
    return updated


# ---------------------------- 4. REMOVE BOTTLENECK ----------------------------
def remove_bottlenecks(data: dict, percentage: float = 0.0) -> dict:
    """
    Mathematically rebalance process times to 'remove' bottlenecks.
    If 10 bottlenecks exist → sum their delay time → redistribute across all steps.
    """
    updated = deepcopy(data)
    nodes = updated.get("process_flow_nodes", [])

    # identify bottlenecks and sum delay
    bottlenecks = [n for n in nodes if n.get("isBottleneck")]
    if not bottlenecks:
        return updated

    total_extra_time = 0.0
    for b in bottlenecks:
        # assume "value" in minutes; convert to seconds for consistency
        total_extra_time += float(b["value"]) * 60.0 * (percentage if percentage else 1)

    avg_increase_sec = total_extra_time / len(nodes) if nodes else 0

    # distribute improvement equally
    for n in nodes:
        new_val_sec = float(n["value"]) * 60.0 + avg_increase_sec
        n["value"] = str(round(new_val_sec / 60.0, 2))
        n["isBottleneck"] = False
        n["descriptions"].append(f"Rebalanced step time (+{round(avg_increase_sec/60,2)}m) to remove bottlenecks.")

    updated["global_metrics"]["Bottleneck_Analysis_Simplified"]["Bottlenecks_Based_on_Avg_Duration_Count"] = 0
    return updated


# ---------------------------- 5. HANDLE DROPOUT ----------------------------
def get_dropout_cases(data: dict) -> dict:
    """Return all dropout steps for visibility."""
    dropouts = [n for n in data.get("process_flow_nodes", []) if n.get("isDropout")]
    return {"dropout_steps": dropouts}


# ---------------------------- 6. MAIN ORCHESTRATOR ----------------------------
def process_user_request(user_input: str, process_data: dict) -> dict:
    """
    Central controller that takes user input and data,
    processes intent, performs calculations, and returns structured output.
    """
    intent = parse_user_input(user_input)

    # If OPENAI_API_KEY is present and openai package is available, ask OpenAI to
    # refine / confirm the parsed intent. This improves handling of ambiguous or
    # short queries (keywords), catches more edge cases, and can supply missing
    # fields (like 'step' or numeric percentages) more reliably than local regexes.
    # The mathematical functionality is unchanged; we only use the AI to improve
    # the detected intent JSON before acting on it.
    try:
        if openai is not None and (os.getenv("OPENAI_API_KEY") or getattr(openai, 'api_key', None)):
            # initialize if needed
            if not getattr(openai, 'api_key', None):
                openai.api_key = os.getenv("OPENAI_API_KEY")

            # Build a small system + user prompt asking for JSON only
            system_prompt = (
                "You are an assistant that extracts structured intent from a user's short "
                "text for a process-mining chatbot. Return ONLY a valid JSON object with the keys: "
                "action (show/remove/reduce/increase), target (loop/bottleneck/dropout/cost), "
                "percentage (floating between 0.0 and 1.0 or null), step (string or null)."
            )

            user_prompt = (
                f"User query: {user_input}\n\n"
                f"Local parsed intent: {json.dumps(intent)}\n\n"
                "If any field is missing or clearly wrong, correct it. Use null for unknown fields."
            )

            # Call ChatCompletion with low temperature for deterministic output
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=250,
            )

            # Support both response.choices[0].message.content and older dict form
            ai_content = None
            try:
                ai_content = resp.choices[0].message.content  # type: ignore
            except Exception:
                try:
                    ai_content = resp.choices[0]["message"]["content"]
                except Exception:
                    ai_content = None

            if ai_content:
                # Attempt to parse JSON from the assistant reply. The assistant is
                # instructed to return only JSON, but be defensive here.
                try:
                    ai_intent = json.loads(ai_content)
                    # Merge: prefer AI extracted values when present (not null)
                    for k in ["action", "target", "percentage", "step"]:
                        if k in ai_intent and ai_intent[k] is not None:
                            intent[k] = ai_intent[k]
                except Exception:
                    # If parsing fails, ignore and continue with local parse
                    pass
    except Exception:
        # If anything goes wrong with the API call, silently continue using local parse
        pass
    action = intent["action"]
    target = intent["target"]
    percentage = intent["percentage"]
    step = intent["step"]

    response = {"user_request": user_input, "intent": intent, "output": None}

    if action == "show" and target:
        response["output"] = get_requested_info(process_data, target)

    elif action == "remove" and target == "loop":
        response["output"] = remove_loops(process_data, step)

    elif action == "remove" and target == "bottleneck":
        response["output"] = remove_bottlenecks(process_data, percentage)

    elif action == "show" and target == "dropout":
        response["output"] = get_dropout_cases(process_data)

    elif action == "reduce" and target == "cost":
        # cost reduction = remove bottlenecks
        response["output"] = remove_bottlenecks(process_data, percentage or 1.0)

    else:
        response["output"] = {"message": "Could not interpret user intent clearly."}

    return response




def save_simulation_result(result: dict, action: str, target: str) -> None:
    """Save simulation result to a new JSON file with descriptive name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_{action}_{target}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved simulation result to {filename}")

if __name__ == '__main__':
    # --------------------------- INITIAL SETUP ---------------------------
    try:
        # Import data directly from process_analysis_output.py
        print("🧠 Loading process mining data...")

        def load_data_from_pyfile(path: str) -> dict:
            """Load a Python file that defines a `data = {...}` structure.
            This helper is defensive: it will replace JavaScript-style literals
            (false/true/null) with Python equivalents before executing so files
            generated by other tools don't crash on import.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            raw = open(path, 'r', encoding='utf-8').read()

            # Replace common JS-style literals with Python ones.
            fixed = re.sub(r"\bfalse\b", "False", raw, flags=re.IGNORECASE)
            fixed = re.sub(r"\btrue\b", "True", fixed, flags=re.IGNORECASE)
            fixed = re.sub(r"\bnull\b", "None", fixed, flags=re.IGNORECASE)

            # Execute the fixed code in a temporary namespace and return 'data'
            ns: dict = {}
            try:
                exec(fixed, ns)
            except Exception as e:
                # If exec fails, surface a helpful error
                raise RuntimeError(f"Failed to execute {path}: {e}")

            if 'data' not in ns:
                raise RuntimeError(f"No 'data' variable found in {path}")
            return ns['data']

        process_data = load_data_from_pyfile('process_analysis_output.py')

        print("✅ Process model loaded successfully. Ready to take user queries.\n")

        # --------------------------- CHATBOT LOOP ---------------------------
        while True:
            # Prompt for user query dynamically
            user_query = input("💬 Enter your request (type 'exit' to quit): ").strip()
            if user_query.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting Process Mining Interactive Mode.")
                break

            # Process the user query using our intelligent intent handler
            response = process_user_request(user_query, process_data)

            # Print and save the result
            print("\n🧾 Updated Process Result:\n")
            print(json.dumps(response, indent=4))

            # Save result if we made changes (not just showing info)
            if response["intent"]["action"] != "show" and response["output"] is not None:
                save_simulation_result(
                    response["output"],
                    response["intent"]["action"] or "unknown",
                    response["intent"]["target"] or "all"
                )

            print("\n──────────────────────────────────────────────\n")

    except Exception as e:
        print(json.dumps({"Error": str(e)}, indent=4))