from .wake_word import WakeWordDetector
from .stt import SpeechToText
import sounddevice as sd
import numpy as np
import time
import os
from dotenv import load_dotenv
import openai

class VoiceAssistant:
    def __init__(self):
        # Load OpenAI API key
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.wake_word_detector = WakeWordDetector()
        self.stt = SpeechToText()
        print("Assistant initialized!")

    def run(self):
        print("Starting voice assistant...")
        print("Listening for wake word 'Hey Austin'...")
        
        try:
            while True:
                if self.wake_word_detector.listen():
                    print("\nWake word detected! Recording your message...")
                    try:
                        audio = self.record_audio()
                        print("Recording complete. Transcribing...")
                        text = self.stt.transcribe(audio)
                        if text:
                            print(f"\nYou said: {text}\n")
                        else:
                            print("Could not transcribe audio")
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                    
                    time.sleep(1)
                    print("Listening for wake word 'Hey Austin'...")
                
        except KeyboardInterrupt:
            print("\nStopping voice assistant...")
        except Exception as e:
            print(f"Error: {e}")

    def record_audio(self, duration=5, sample_rate=16000):
        print("Recording started...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording finished.")
        return audio

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()