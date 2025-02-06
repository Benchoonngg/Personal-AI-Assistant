import os
import logging
from TTS.api import TTS
import sounddevice as sd
import numpy as np

class TextToSpeech:
    def __init__(self):
        # Suppress TTS initialization messages
        logging.getLogger('TTS').setLevel(logging.ERROR)
        os.environ['TTS_VERBOSE'] = '0'
        
        print("Initializing TTS...")
        # Choose one of these models:
        # 1. "tts_models/en/jenny/jenny" - Natural female voice (~300MB)
        # 2. "tts_models/en/ljspeech/tacotron2-DDC" - Classic female voice (~70MB)
        # 3. "tts_models/en/vctk/vits" - Multiple speakers (~100MB)
        self.tts = TTS(
            model_name="tts_models/en/jenny/jenny",
            progress_bar=True
        )
        print("TTS ready!")

    def speak(self, text):
        try:
            print("Generating speech...")
            # Generate audio wave with increased speed
            wav = self.tts.tts(text, speed=1.8)
            
            # Convert to float32 and normalize
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Play the audio
            print("Playing audio...")
            sd.play(audio_data, samplerate=22050)
            sd.wait()
            
        except Exception as e:
            print(f"Error in TTS: {e}")