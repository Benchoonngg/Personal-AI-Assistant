from .wake_word import WakeWordDetector
from .stt import SpeechToText
from .tts import TextToSpeech
import sounddevice as sd
import numpy as np
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime

class VoiceAssistant:
    def __init__(self):
        # Load OpenAI API key
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load assistant configuration
        self.load_config()
        
        self.wake_word_detector = WakeWordDetector()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.silence_threshold = 500
        
        # Initialize conversation with system prompt
        self.conversation_history = [self.config["system_prompt"]]
        
        # Create history directory if it doesn't exist
        os.makedirs("history", exist_ok=True)
        
        # Generate unique filename for this session
        self.conversation_file = f"history/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print("Assistant initialized!")

    def load_config(self):
        try:
            config_path = "config/assistant_config.json"  # Make sure this path is correct
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                print(f"Configuration loaded successfully from {config_path}")
                # Debug print to verify loaded content
                print(f"System prompt: {self.config['system_prompt']['content'][:50]}...")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Fallback to default configuration
            self.config = {
                "system_prompt": {
                    "role": "system",
                    "content": "Keep Responses short please."
                },
                "model": "gpt-4-turbo-preview",
                "temperature": 0.7
            }
            print("Using default configuration")

    def process_with_gpt(self, text):
        try:
            self.conversation_history.append({"role": "user", "content": text})
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=self.conversation_history,
                temperature=self.config["temperature"]
            )
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Debug print to confirm model
            print(f"Using model: {response.model}")
            
            return assistant_message
        except Exception as e:
            print(f"Error with GPT processing: {e}")
            return None

    def save_conversation(self):
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "conversation": self.conversation_history
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def run(self):
        print("Starting voice assistant...")
        print("Listening for wake word 'Hey Austin'...")
        
        try:
            while True:
                if self.wake_word_detector.listen():
                    print("\nWake word detected! Listening to your message...")
                    while True:  # Continue conversation until long silence
                        try:
                            audio = self.record_until_silence()
                            if audio is not None:
                                print("Recording complete. Transcribing...")
                                text = self.stt.transcribe(audio)
                                if text:
                                    print(f"\nYou said: {text}")
                                    
                                    # Process with GPT
                                    print("\nProcessing with AI...")
                                    ai_response = self.process_with_gpt(text)
                                    if ai_response:
                                        print(f"\nAI Response: {ai_response}")
                                        # Convert response to speech using Coqui TTS
                                        print("Converting to speech...")
                                        self.tts.speak(ai_response)
                                        self.save_conversation()  # Save after each response
                                    
                                    print("\nListening for your next message... (or wait for timeout)")
                                else:
                                    print("No speech detected, returning to wake word mode...")
                                    break
                            else:
                                print("Conversation ended due to silence")
                                break
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            break
                    
                    print("\nListening for wake word 'Hey Austin'...")
                
        except KeyboardInterrupt:
            print("\nSaving conversation and stopping...")
            self.save_conversation()
        except Exception as e:
            print(f"Error: {e}")
            self.save_conversation()

    def record_until_silence(self, sample_rate=16000, silence_duration=2):
        print("Recording started...")
        silence_threshold = 500  # Adjust this value based on testing
        silence_frames = 0
        max_silence_frames = int(silence_duration * sample_rate)
        
        # Initialize an empty array for storing audio
        audio_chunks = []
        
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
                while True:
                    audio_chunk, _ = stream.read(int(sample_rate/4))  # Read 0.25 seconds of audio
                    audio_chunks.append(audio_chunk)
                    
                    # Check for silence
                    if np.max(np.abs(audio_chunk)) < silence_threshold:
                        silence_frames += int(sample_rate/4)
                        print(f"Silence detected: {silence_frames/sample_rate:.1f}s")  # Debug print
                        if silence_frames >= max_silence_frames:
                            print("Silence threshold reached")
                            break
                    else:
                        silence_frames = 0
                        
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
            
        if not audio_chunks:
            return None
            
        # Combine all audio chunks
        audio = np.concatenate(audio_chunks)
        print("Recording finished.")
        return audio

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()