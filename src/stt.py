import whisper
import numpy as np

class SpeechToText:
    def __init__(self, model_name="tiny.en"):
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_name)
        print("Whisper model loaded")

    def transcribe(self, audio):
        try:
            # Convert audio to float32 and normalize
            audio = audio.flatten().astype(np.float32) / 32768.0
            result = self.model.transcribe(audio, fp16=False)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""