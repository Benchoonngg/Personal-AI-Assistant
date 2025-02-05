from TTS.api import TTS
import sounddevice as sd
import numpy as np

class TextToSpeech:
    def __init__(self):
        print("Initializing Coqui TTS...")
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        print("TTS model loaded")

    def speak(self, text):
        try:
            print("Generating speech...")
            # Generate audio wave
            wav = self.tts.tts(text)
            
            # Convert to float32 and normalize
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Play the audio
            print("Playing audio...")
            sd.play(audio_data, samplerate=22050)
            sd.wait()
            
        except Exception as e:
            print(f"Error in TTS: {e}")