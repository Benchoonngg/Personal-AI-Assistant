# voice cloning
# clone voice from wav file
# clones everytime you run the script

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
        # You can choose from these models:
        # 1. "tts_models/en/jenny/jenny" - More natural female voice
        # 2. "tts_models/en/vctk/vits" - Multiple speaker voices
        # 3. "tts_models/multilingual/multi-dataset/xtts_v2" - More expressive, natural voice
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        # Reference voice file (WAV format recommended)
        self.reference_wav = "path/to/your_voice.wav"
        print("TTS ready!")

    def speak(self, text):
        try:
            print("Generating speech...")
            # Clone voice on-the-fly
            wav = self.tts.tts(
                text=text,
                speaker_wav=self.reference_wav,
                language="en",
                speed=1.8
            )
            
            # Convert to float32 and normalize
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Play the audio
            print("Playing audio...")
            sd.play(audio_data, samplerate=22050)
            sd.wait()
            
        except Exception as e:
            print(f"Error in TTS: {e}")