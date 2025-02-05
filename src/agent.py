from .wake_word import WakeWordDetector
from .stt import SpeechToText
from .tts import TextToSpeech
import sounddevice as sd
import numpy as np
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

class VoiceAssistant:
    def __init__(self):
        # Load OpenAI API key
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.wake_word_detector = WakeWordDetector()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.silence_threshold = 500  # Adjust this value based on testing
        print("Assistant initialized!")

    def process_with_gpt(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with GPT processing: {e}")
            return None

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
            print("\nStopping voice assistant...")
        except Exception as e:
            print(f"Error: {e}")

    def record_until_silence(self, sample_rate=16000, silence_duration=1):
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