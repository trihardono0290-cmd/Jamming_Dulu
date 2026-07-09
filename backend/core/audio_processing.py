import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Parameters for audio processing
SR = 22050               # Default sample rate
N_FFT = 4096             # Meningkatkan resolusi frekuensi (dari 2048 ke 4096) agar nada gitar lebih presisi
HOP_LENGTH = 512         # Jarak antar frame FFT
N_MELS = 256             # Meningkatkan jumlah Mel bands (dari 128 ke 256) untuk detail frekuensi lebih rapat
FMIN = 50                # Batas bawah gitar (dari 20 ke 50 Hz - Low E gitar adalah ~82 Hz)
FMAX = 3000              # Batas atas gitar (dari 8000 ke 3000 Hz - harmoni penting gitar tidak melebihi 3kHz)
TARGET_SIZE = (128, 128) # Final image size (width, height)

def generate_mel_spectrogram(audio_signal, sr=SR):
    """
    Core function to generate Mel-Spectrogram with Logarithmic Compression.
    Returns a normalized 2D numpy array [0, 1].
    """
    # 1. Generate Mel-Spectrogram (Not linear spectrogram)
    # Skala Mel digunakan untuk memadatkan frekuensi tinggi yang kurang penting bagi musik
    mel_spec = librosa.feature.melspectrogram(
        y=audio_signal, 
        sr=sr, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # 2. Logarithmic Compression (Power-to-dB)
    # Wajib untuk memastikan energi petikan yang terlalu keras tidak mendominasi gambar spektrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Normalize intensity to [0, 1] for CNN input
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_normalized

def process_audio_realtime(audio_data, sr=SR):
    """
    Used in BACKEND INFERENCE (FastAPI) for real-time processing.
    Avoids using matplotlib or disk I/O for minimum latency.
    
    Args:
        audio_data: 1D numpy array containing the raw audio signal.
    Returns:
        numpy array of shape (128, 128, 1) ready for CNN prediction.
    """
    # Get normalized spectrogram
    mel_spec_normalized = generate_mel_spectrogram(audio_data, sr)
    
    # Resize to target size (128x128) using PIL for fast memory operation
    # Explicitly define mode='L' for grayscale 8-bit pixels
    mel_spec_img = Image.fromarray(np.uint8(mel_spec_normalized * 255), mode='L')
    mel_spec_resized = mel_spec_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    # Convert back to numpy array and normalize back to [0, 1]
    final_array = np.array(mel_spec_resized, dtype=np.float32) / 255.0
    
    # Repeat the grayscale channel 3 times for MobileNetV2 (RGB input) shape -> (128, 128, 3)
    final_array = np.stack([final_array]*3, axis=-1)
    
    return final_array

def process_and_save_for_training(wav_path, output_path):
    """
    Used OFFLINE for DATASET PREPARATION.
    Reads a .wav file and saves the 128x128 grayscale spectrogram image.
    """
    try:
        # Load audio file
        audio_signal, sr = librosa.load(wav_path, sr=SR)
        
        # Get normalized spectrogram
        mel_spec_normalized = generate_mel_spectrogram(audio_signal, sr)
        
        # Resize using PIL
        mel_spec_img = Image.fromarray(np.uint8(mel_spec_normalized * 255), mode='L')
        mel_spec_resized = mel_spec_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as grayscale image
        mel_spec_resized.save(output_path)
        print(f"Saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return False

# Example usage/test execution
if __name__ == "__main__":
    # Dry run test to ensure syntax and basic logic flow works
    print("Running dry-run test for audio processing...")
    dummy_audio = np.random.uniform(-1, 1, SR * 2) # 2 seconds of random noise
    
    # Test realtime function
    cnn_input = process_audio_realtime(dummy_audio, SR)
    print(f"Realtime output shape: {cnn_input.shape}") # Expected (128, 128, 1)
    print(f"Realtime output max: {cnn_input.max():.2f}, min: {cnn_input.min():.2f}")
