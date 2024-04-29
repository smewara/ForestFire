import librosa
import numpy as np
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class MelProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__(n_fft=n_fft, hop_length=hop_length)

    def compute_segmented_spectrograms(self, audio_path, kind_of_augmentation = None, duration_in_sec=2.5):
        # Load audio file
        y, sr = super().normalize_audio(audio_path=audio_path, kind_of_augmentation=kind_of_augmentation)
        
        #The load method returns two variables, the time series (y) and the sample rate (sr), which is the number of samples per second.

        # Split audio into segments
        segments = super().split_audio_into_segments(y=y, sr=sr, duration=duration_in_sec, overlap=0.5)

        # Compute Mel spectrograms from segments      
        mel_spectrogram = self._compute_spectrogram_from_segments(segments)
        
        return mel_spectrogram
    
    def _compute_spectrogram_from_segments(self, segments):
        mel_spectrogram = []

        # Compute Mel for each segment
        for start_time, segment in segments:
            mel = librosa.feature.melspectrogram(y=segment, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(mel)
            mel_spectrogram.append((magnitude, start_time))

        return mel_spectrogram
    
    def compute_spectrogram(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path)
        return np.abs(librosa.feature.melspectrogram(y=y, n_fft=self.n_fft, hop_length=self.hop_length))

