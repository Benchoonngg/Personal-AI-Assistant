import pvporcupine
import sounddevice as sd
import numpy as np

class WakeWordDetector:
    def __init__(self, wake_word_path="config/wake_words/Hey-Austin_en_mac_v3_0_0.ppn"):
        print("Initializing wake word detector...")
        self.porcupine = pvporcupine.create(
            access_key="P93SpbCuwK3lWXL0gm5nQhwZC4trwMY0q5Q7o0V6x5PY7GmnbWJScQ==",  # Replace with your actual AccessKey
            keyword_paths=[wake_word_path]
        )
        print(f"Sample rate: {self.porcupine.sample_rate}")
        print(f"Frame length: {self.porcupine.frame_length}")
        
        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=self.porcupine.sample_rate,
            blocksize=self.porcupine.frame_length,
            dtype=np.int16
        )
        self.audio_stream.start()
        print("Audio stream started")

    def listen(self):
        try:
            audio_data, _ = self.audio_stream.read(self.porcupine.frame_length)
            pcm = audio_data.flatten()
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
            return keyword_index >= 0
        except Exception as e:
            print(f"Error in listen(): {e}")
            return False 