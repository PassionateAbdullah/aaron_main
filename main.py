from Landing_page_bot_2 import ChatSession


def main():
    """Run CLI chatbot."""
    print("🤖 Landing Page Chatbot (FAISS + Gemini)\n")

    try:
        session = ChatSession()
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Have a productive day! 👋")
                break

            if user_input.lower() == "reload":
                session.reload_kb()
                continue

            if not user_input:
                continue

            response = session.send_message(user_input)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\nBot: Have a productive day! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()