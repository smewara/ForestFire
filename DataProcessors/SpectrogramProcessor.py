import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

class SpectrogramProcessor:
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def normalize_and_augment_audio(self, audio_path, target_loudness=-10, low_pass_freq=30000):
        os.makedirs(output_dir, exist_ok=True)
        # Load audio file
        audio, sr = librosa.load(audio_path, sr = 16000)

        # Apply normalisation and filter to all
        
        # Normalize audio
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(y)
        normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)
    
        # Low-pass filter
        normalized_cutoff_frequency = low_pass_freq / (0.5 * sr)
        sos = butter(10, normalized_cutoff_frequency, 'lp', fs=sr, output='sos')
        augmented_audio = sosfilt(sos, normalized_audio)

        # Choose augmentation (comment out others)

        # Pitch shift
        pitch_shift_factor = np.random.uniform(-2, 2)
        augmented_audio = librosa.effects.pitch_shift(filtered_audio, sr=16000, n_steps=pitch_shift_factor)

        ## Time stretch
        #stretch_rate = np.random.uniform(0.8, 1.2)
        #augmented_audio = librosa.effects.time_stretch(filtered_audio, rate=stretch_rate)

        ## Volume adjustment
        #volume_adjustment = np.random.uniform(0.2, 0.8)
        #augmented_audio = filtered_audio * volume_adjustment
              
        return augmented_audio, sr
    
    def split_audio_into_segments(self, y, sr, duration=2.5, overlap=0.5):
        segment_samples = int(duration * sr)
        hop_length = int(segment_samples * (1 - overlap))

        # Split audio into overlapping segments
        segments = []
        start_sample = 0
        while start_sample + segment_samples < len(y):
            segment = y[start_sample:start_sample + segment_samples]
            segments.append((start_sample / sr, segment))
            start_sample += hop_length

        return segments

    def compute_spectrogram(self, audio_path):
        raise NotImplementedError("Subclasses must implement compute_spectrogram method")

    def compute_segmented_spectrograms(self, audio_path, duration_in_sec=2.5):
        raise NotImplementedError("Subclasses must implement compute_spectrogram method")

    def save_spectrogram(self, spectrograms, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)

        for index, (spectrogram, start_time) in enumerate(spectrograms):
            segment_filename = f"{filename}_{index}"
            np.save(os.path.join(output_dir, segment_filename), spectrogram)

    def plot_spectrogram(self, spectrogram, title, sr, hop_length, y_axis='log'):
        y_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        plt.figure(figsize=(25,10))
        librosa.display.specshow(y_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(title)
        plt.show()

    def plot_scalogram(self, audio, scalogram, freq, title, sr):
        plt.subplot(1, 4, 3)
        plt.imshow(scalogram, extent=[0, len(audio)/sr, freq[-1], freq[0]], aspect='auto', cmap='magma', interpolation='bilinear', norm='log')
        plt.colorbar(label='Intensity')
        plt.xlabel('Time (s)')
        plt.ylabel('Scale')
        plt.title(title)
        plt.show()

   
