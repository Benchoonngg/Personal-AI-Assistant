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
        # Male voice options:
        # 1. "tts_models/en/vctk/vits" - Multiple speakers including male voices
        # 2. "tts_models/en/ljspeech/glow-tts" - Deeper voice
        # 3. "tts_models/en/sam/tacotron-DDC" - Clear male voice
        self.tts = TTS(
            model_name="tts_models/en/vctk/vits",
            progress_bar=True
        )
        # Set male speaker (VCTK has multiple speakers)
        self.speaker = "p226"  # Male speaker code | can use p225 for a deeper voice | p224 for a higher pitch 
        print("TTS ready!")

    def speak(self, text):
        try:
            print("Generating speech...")
            # Generate audio wave with increased speed and male voice
            wav = self.tts.tts(
                text=text,
                speaker=self.speaker,  # Specify male speaker
                speed=1.2  # Increased speed from 1.8 to 2.0
            )
            
            # Convert to float32 and normalize
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Play the audio with higher sample rate for clarity
            print("Playing audio...")
            sd.play(audio_data, samplerate=24000)  # Increased from 22050
            sd.wait()
            
        except Exception as e:
            print(f"Error in TTS: {e}")