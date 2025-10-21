"""
Landing Page Chatbot - Clean Function-Based Version
Simple, modular functions that work independently.
"""

import os
import json
import re
from difflib import SequenceMatcher
import google.generativeai as genai
from dotenv import load_dotenv
import random

# ===================== CONFIGURATION =====================

def init_config(api_key=None, kb_path="landing.json", model="gemini-2.5-flash"):
    """Initialize and return configuration dict."""
    load_dotenv()
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in environment")
    
    genai.configure(api_key=api_key)
    
    return {
        "kb_path": kb_path,
        "model": model,
        "max_history": 5,
        "similarity_threshold": 6.0,
        "fuzzy_threshold": 7.0
    }


# ===================== KNOWLEDGE BASE =====================

def load_knowledge_base(kb_path):
    """Load and normalize knowledge base from JSON file."""
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different formats
    entries = data["data"] if isinstance(data, dict) and "data" in data else data
    
    # Normalize entries
    kb = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("question"):
            kb.append({
                "question": entry["question"].strip().lower(),
                "answer": entry.get("answer", "").strip()
            })
        elif isinstance(entry, str) and entry.strip():
            kb.append({"question": entry.strip().lower(), "answer": ""})
    
    return kb


def save_knowledge_base(kb, kb_path):
    """Save knowledge base to JSON file."""
    data = {"data": [{"question": e["question"], "answer": e["answer"]} for e in kb]}
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_kb_entry(kb, question, answer):
    """Add new entry to knowledge base."""
    kb.append({"question": question.strip().lower(), "answer": answer.strip()})
    return kb


# ===================== SIMILARITY =====================

def fuzzy_score(text1, text2):
    """Calculate fuzzy similarity score (0-10)."""
    if not text1 or not text2:
        return 0.0
    ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio * 10.0


def extract_score(text):
    """Extract numeric score (0-10) from text."""
    if not text:
        return None
    match = re.search(r"([0-9](?:\.[0-9])?)", text)
    if match:
        try:
            val = float(match.group(1))
            return val if 0.0 <= val <= 10.0 else None
        except:
            return None
    return None


# ===================== AI INTERFACE =====================

def rate_similarity_ai(query, candidate, model_name):
    """Use AI to rate similarity between query and candidate."""
    model = genai.GenerativeModel(model_name)
    prompt = (
        f"Rate similarity from 0 to 10 (only output the number):\n\n"
        f"User Query: {query}\n"
        f"KB Question: {candidate}\n\n"
        f"Respond with a single number between 0 and 10."
    )
    
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        score = extract_score(text)
        return score if score is not None else fuzzy_score(query, candidate)
    except:
        return fuzzy_score(query, candidate)


def generate_ai_response(query, history, model_name):
    """Generate AI response when no KB match found."""
    model = genai.GenerativeModel(model_name)
    
    # Format history
    history_text = ""
    if history:
        parts = [f"User: {h['user']}\nBot: {h['bot']}" for h in history[-5:]]
        history_text = f"Recent conversation:\n" + "\n".join(parts) + "\n\n"
    
    prompt = (
        f"You are a helpful business process assistant. "
        f"{history_text}"
        f"User asked: '{query}'. "
        f"Answer based on KPI/process analytics context. "
        f"If unsure, politely suggest related topics."
    )
    
    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "I don't have an answer right now. Can you rephrase?"
    except:
        return "I don't have this answer right now. Please try again."


# ===================== MATCHING =====================

def find_best_match(query, kb, model_name, fuzzy_threshold=7.0):
    """Find best matching KB entry for query."""
    query_clean = query.strip().lower()
    
    if not query_clean or not kb:
        return None, 0.0
    
    # Calculate fuzzy scores
    scored = [(fuzzy_score(query_clean, e["question"]), e) for e in kb]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_fuzzy_score, best_entry = scored[0]
    
    # Return if fuzzy score is good enough
    if best_fuzzy_score >= fuzzy_threshold:
        return best_entry, round(best_fuzzy_score, 2)
    
    # Use AI to rate top 5 candidates
    top_candidates = [e for _, e in scored[:5]]
    best_match, best_score = None, -1.0
    
    for candidate in top_candidates:
        score = rate_similarity_ai(query, candidate["question"], model_name)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match, round(best_score, 2)


# ===================== GREETING =====================

def is_greeting(message):
    """Check if message is a greeting."""
    keywords = ["hi", "hello", "hey", "good morning", "good afternoon", 
                "good evening", "how are you", "help", "what can you do"]
    return any(kw in message.lower() for kw in keywords)


def get_greeting_response():
    """Get random greeting response."""
    responses = [
        "Hello! 👋 I'm your KPI and Process Analysis Assistant. How can I help?",
        "Good day! I specialize in KPIs, process mining, and performance insights.",
        "Hi there! 😊 I can help you explore KPI metrics and process analytics.",
        "Welcome! I assist with KPI-based process analysis and benchmarking.",
        "Hello! Let's talk about your business process performance."
    ]
    return random.choice(responses)


# ===================== CONVERSATION =====================

def add_to_history(history, user_msg, bot_msg, max_history=5):
    """Add conversation turn to history."""
    history.append({"user": user_msg, "bot": bot_msg})
    if len(history) > max_history:
        history.pop(0)
    return history


def build_context_query(current_query, history):
    """Build context-aware query from history."""
    if history and history[-1].get("user"):
        return f"{history[-1]['user']} {current_query}"
    return current_query


# ===================== MAIN CHAT FUNCTION =====================

def chat(user_input, config=None, history=None):
    """
    Main chat function - processes user input and returns response.
    
    Args:
        user_input: User's message
        config: Config dict (creates default if None)
        history: Conversation history list (creates new if None)
    
    Returns:
        Bot's response string
    """
    # Initialize
    config = config or init_config()
    history = history if history is not None else []
    
    # Load KB
    kb = load_knowledge_base(config["kb_path"])
    
    # Handle greetings
    if is_greeting(user_input):
        response = get_greeting_response()
        add_to_history(history, user_input, response, config["max_history"])
        return response
    
    # Build context query
    context_query = build_context_query(user_input, history)
    
    # Find match
    match, score = find_best_match(context_query, kb, config["model"], config["fuzzy_threshold"])
    
    # Return KB answer if found
    if match and score >= config["similarity_threshold"]:
        answer = match["answer"] or "I'm not sure, but I can find out more."
        response = f"{answer}\n\nAnything else about KPIs or process improvement?"
        add_to_history(history, user_input, response, config["max_history"])
        return response
    
    # Generate AI fallback
    response = generate_ai_response(user_input, history, config["model"])
    add_to_history(history, user_input, response, config["max_history"])
    return response


# ===================== CONVENIENCE WRAPPER =====================

# Global state for backward compatibility
_global_config = None
_global_history = []

def chat_simple(user_input, kb_path="landing.json", model="gemini-2.5-flash"):
    """Simple chat function with global state (backward compatible)."""
    global _global_config, _global_history
    
    if _global_config is None:
        _global_config = init_config(kb_path=kb_path, model=model)
    
    return chat(user_input, _global_config, _global_history)


# ===================== CLI =====================

def main():
    """Run CLI chatbot."""
    print("🤖 Landing Page Chatbot\n")
    
    config = init_config()
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Have a productive day! 👋")
                break
            
            if not user_input:
                continue
            
            response = chat(user_input, config, history)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nBot: Have a productive day! 👋")
            break


if __name__ == "__main__":
    main()