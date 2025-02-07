# Training script (separate file)
import os
from TTS.trainer import Trainer
from TTS.config import load_config
from TTS.utils.manage import ModelManager
import torch

def prepare_dataset(voice_samples_dir):
    """Prepare dataset metadata"""
    metadata = []
    for file in os.listdir(voice_samples_dir):
        if file.endswith('.wav'):
            # We need actual transcriptions for each audio file
            # Current code uses placeholder text which won't work well
            metadata.append(f"{file}|Your transcribed text here|your_voice|")
    
    # Save metadata
    with open('metadata.txt', 'w', encoding='utf-8') as f:
        for line in metadata:
            f.write(line + '\n')

def validate_audio(file_path):
    """Validate audio files meet requirements"""
    try:
        import soundfile as sf
        data, samplerate = sf.read(file_path)
        
        if samplerate != 16000:
            print(f"Warning: {file_path} sample rate is {samplerate}Hz, needs to be 16000Hz")
            return False
            
        if len(data.shape) > 1:
            print(f"Warning: {file_path} is not mono")
            return False
            
        duration = len(data) / samplerate
        if duration < 1 or duration > 20:
            print(f"Warning: {file_path} duration ({duration}s) is not optimal")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False

def train_tts_model():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: GPU not detected. Training might be slow.")
    
    # Training configuration
    config = {
        "model": "xtts_v2",
        "batch_size": 8,
        "eval_batch_size": 8,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 4,
        "run_eval": True,
        "test_delay_epochs": -1,
        "epochs": 1000,
        "text_cleaner": "multilingual_cleaners",
        "use_phonemes": True,
        "phoneme_language": "en-us",
        "output_path": "training_output",
        "datasets": [
            {
                "name": "your_voice_dataset",
                "path": "path/to/your_voice_samples",
                "meta_file_train": "metadata.txt",
                "language": "en",
                "audio": {
                    "sample_rate": 22050,
                    "normalize": True
                }
            }
        ],
        # XTTS v2 specific settings
        "model_args": {
            "use_speaker_embedding": True,
            "speaker_embedding_channels": 512,
            "use_language_embedding": True,
            "language_embedding_channels": 256,
            "encoder_sample_rate": 16000,
            "encoder_in_channels": 1
        }
    }

    # Initialize trainer
    trainer = Trainer(
        config,
        output_path="tts_trainer/fine-tuned-voices",
        gpu=0 if torch.cuda.is_available() else None
    )

    # Add callbacks for progress monitoring
    trainer.add_callback(
        "on_epoch_end",
        lambda trainer: print(f"Epoch {trainer.current_epoch}: Loss = {trainer.avg_loss}")
    )
    
    # Add model checkpointing
    trainer.add_callback(
        "on_save_checkpoint",
        lambda trainer: print(f"Saved checkpoint at epoch {trainer.current_epoch}")
    )

    # Start training
    try:
        print("Starting training...")
        trainer.fit()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")

def main():
    # Directory containing your voice samples
    VOICE_SAMPLES_DIR = "tts_trainer/voice_samples"
    
    # Create necessary directories
    os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
    os.makedirs("training_output", exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    prepare_dataset(VOICE_SAMPLES_DIR)
    
    # Start training
    print("Initializing training...")
    train_tts_model()

if __name__ == "__main__":
    main()

# Usage after training
class TextToSpeech:
    def __init__(self):
        self.tts = TTS(model_path="path/to/your_trained_model")
    
    def speak(self, text):
        wav = self.tts.tts(text)

    def speak(self, text):
        try:
            print("Generating speech...")
            # Generate audio wave with increased speed
            # Adjust speed value: 1.0 is normal, 2.0 is twice as fast
            wav = self.tts.tts(text, speed=1.8)  # Increased speed from 1.5 to 1.8
            
            # Convert to float32 and normalize
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Play the audio
            print("Playing audio...")
            sd.play(audio_data, samplerate=22050)
            sd.wait()
            
        except Exception as e:
            print(f"Error in TTS: {e}")