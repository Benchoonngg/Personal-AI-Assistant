from src.agent import VoiceAssistant
import logging

def main():
    try:
        print("Starting Personal AI Assistant...")
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\nShutting down assistant...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()