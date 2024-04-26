import os
import librosa
import pywt
import numpy as np
import soundfile as sf
import pyloudnorm
from scipy.signal import resample, butter, sosfilt
from librosa import effects
import uuid


def normalize_and_augment_audio(audio_file, output_dir, target_loudness=-23, low_pass_freq=20000):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    # Check if audio length is sufficient for processing
    # if len(y) < 44100:
    #     print(f"Skipping {filename} due to insufficient length.")
    #     return []
    
    # Normalize audio
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(y)
    normalized_audio = pyloudnorm.normalize.loudness(y, loudness, target_loudness)

    # Low-pass filter
    normalized_cutoff_frequency = low_pass_freq / (0.5 * sr)
    sos = butter(10, normalized_cutoff_frequency, 'lp', fs=sr, output='sos')
    filtered_audio = sosfilt(sos, normalized_audio)

    # Save normalized and high-pass filtered audio
    # output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_normalized.wav")
    # sf.write(output_filename, filtered_audio, sr)
    # print(f"Saved normalized and high-pass filtered audio: {output_filename}")

    # Generate augmented versions
    augmented_audio = [filtered_audio]

    if np.random.rand() < 0.2:
        # Pitch shift
        pitch_shift_factor = np.random.uniform(-2, 2)
        pitch_shifted_audio = librosa.effects.pitch_shift(filtered_audio, sr=16000, n_steps=pitch_shift_factor)
        augmented_audio.append(pitch_shifted_audio)
        # print("Pitch shift")

    if np.random.rand() < 0.2:
        # Time stretch
        stretch_rate = np.random.uniform(0.8, 1.2)
        time_stretched_audio = librosa.effects.time_stretch(filtered_audio, rate=stretch_rate)
        augmented_audio.append(time_stretched_audio)
        # print("Time stretch")

    if np.random.rand() < 0.2:
        # Volume adjustment
        volume_adjustment = np.random.uniform(0.1, 0.8)
        volume_adjusted_audio = filtered_audio * volume_adjustment
        augmented_audio.append(volume_adjusted_audio)
        # print("Volume adjustment")
    

    #Save augmented audio files
    # for i, audio in enumerate(augmented_audio):
    #     output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_augmented_{i}.wav")
    #     sf.write(output_filename, audio, sr)
    #     print(f"Saved augmented audio: {output_filename}")

    # return None
    # print(len(augmented_audio))
    return augmented_audio

def split_audio_into_segments(audio, sample_rate, duration=2.5, overlap=0.5, min_segment_length=2.5):
    segment_samples = int(duration * sample_rate)
    min_segment_samples = int(min_segment_length * sample_rate)
    hop_length = int(segment_samples * (1 - overlap))

    # Split audio into segments
    segments = []
    start_sample = 0

    while start_sample + segment_samples < len(audio):
        segment = audio[start_sample:start_sample + segment_samples]
        if len(segment) >= min_segment_samples:
            segments.append((segment, start_sample / sample_rate))
        start_sample += hop_length

    return segments

def compute_cwt_scalogram(audio_file, output_dir, filename, wavelet='morl', num_scales=64, target_loudness=-23, target_size=(64, 512)):
    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    # Normalize audio
    augmented_audio_list = normalize_and_augment_audio(audio_file, output_dir, target_loudness=-23, low_pass_freq=100)

    for idx, augmented_audio in enumerate(augmented_audio_list):
        # Split augmented audio into segments
        segments = split_audio_into_segments(augmented_audio, sr)

        for segment, start_time in segments:
            
            segment_scalogram, _ = compute_cwt_scalogram_single(segment, sr, wavelet, num_scales, target_size)
            unique_id = str(uuid.uuid4())[:8]
            
            # Construct the filename with the unique identifier
            segment_filename = f"{os.path.splitext(filename)[0]}_{idx}_{start_time:.2f}_{unique_id}"
            
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{segment_filename}_scalogram.npy"), segment_scalogram)

    return None


def compute_cwt_scalogram_single(audio, sample_rate, wavelet='morl', num_scales=64, target_size=(64, 512)):
    scales = np.linspace(1, 64, num_scales, dtype=int)
    coeffs, freqs = pywt.cwt(audio, scales, wavelet)
    scalogram = np.abs(coeffs) ** 2
    
    # Zero-pad the scalogram to a fixed size
    padded_scalogram = np.zeros((num_scales, coeffs.shape[1]))
    padded_scalogram[:scalogram.shape[0], :scalogram.shape[1]] = scalogram
    
    # Normalize the scalogram
    padded_scalogram = (padded_scalogram - padded_scalogram.min()) / (padded_scalogram.max() - padded_scalogram.min())
    
    # Resample the scalogram to the target size
    resampled_scalogram = resample(padded_scalogram, target_size[0], axis=0)
    resampled_scalogram = resample(resampled_scalogram, target_size[1], axis=1)
    
    return resampled_scalogram, freqs

input_dirs = {
    #'Rainforest': 'Original audio/Rainforest sounds',
    #'Fire': 'Original audio/Fire sounds',
    'Fire': 'Original audio/Rainforest sounds'
}

output_dirs = {
   # 'Rainforest': 'CWT_scalograms2//RAINFOREST',
   # 'Fire': 'CWT_scalograms/2/FIRE',
    'Fire': 'Scalograms/No Fire/Rainforest'
    # 8461 y2m + 8867 fire = 17338
    # 3900 environment + 18011 rainforest = 21911
}   # 17338 - 3900 = 13438

for category, input_dir in input_dirs.items():
    output_dir = output_dirs[category]
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            audio_file = os.path.join(input_dir, filename)
            compute_cwt_scalogram(audio_file, output_dir, os.path.splitext(filename)[0])