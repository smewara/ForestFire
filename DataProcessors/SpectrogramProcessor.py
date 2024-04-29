import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm

class SpectrogramProcessor:
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def normalize_loudness(self, audio, sr):
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, -10)
        return normalized_audio

    def pitch_shift(self, audio, sr):
        # Pitch shift
        pitch_shift_factor = np.random.uniform(-2, 2)
        pitch_shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_factor)
        return pitch_shifted_audio, sr

    def time_stretch(self, audio, sr):
        # Time stretch
        stretch_rate = np.random.uniform(0.8, 1.2)
        time_stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
        return time_stretched_audio, sr
    
    def volume_adjusted(self, audio, sr):
        volume_adjustment = np.random.uniform(0.2, 0.8)
        augmented_audio = audio * volume_adjustment
        return audio, sr

    def normalize_audio(self, audio_path, kind_of_augmentation = None, target_loudness=-23.0):
        # Load audio file
        audio, sr = librosa.load(audio_path, sr = 16000)

        if kind_of_augmentation is not None:
           audio = self.normalize_loudness(audio=audio, sr=sr)
           
        if kind_of_augmentation == 'PitchShifted':
            return self.pitch_shift(audio, sr)
        elif kind_of_augmentation == 'Superimposed':
            return audio, sr
        elif kind_of_augmentation == 'VolumeAdjusted':
            return self.volume_adjusted(audio, sr)
        elif kind_of_augmentation == 'TimeStretched':
            return self.time_stretch(audio, sr)

        return audio, sr
    
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

   
