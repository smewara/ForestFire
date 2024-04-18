import librosa
import os
import numpy as np
import pyloudnorm
import matplotlib.pyplot as plt
import soundfile as sf

class SpectrogramProcessor:
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def normalize_audio(self, audio_path, sample_rate, target_loudness=-23):
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=sample_rate)
    
        # Measure loudness
        meter = pyloudnorm.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
    
        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    
        # Normalize audio
        normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)
        return normalized_audio


    
    def split_audio_into_segments(self, y, sr, duration=3, overlap=0.5):
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

    def compute_segmented_spectrograms(self, audio_path):
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

   
