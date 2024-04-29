import librosa
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor
import numpy as np

class STFTProcessor(SpectrogramProcessor):
    def init(self, n_fft=2048, hop_length=512):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        
    def compute_segmented_spectrograms(self, audio_path, kind_of_augmentation = None, duration_in_sec=2.5):
        # Load audio file
        y, sr = super().normalize_audio(audio_path=audio_path, kind_of_augmentation=kind_of_augmentation)
        
        # Split audio into segments
        segments = super().split_audio_into_segments(y=y, sr=sr, duration=duration_in_sec, overlap=0.5)

        # Compute STFT spectrograms from segments
        spectrograms = self._compute_spectrogram_from_segments(segments)
        
        return spectrograms

    def _compute_spectrogram_from_segments(self, segments):    
        spectrograms = []
        
        # Compute STFT for each segment       
        for start_time, segment in segments: 
            stft = librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_length)  
            magnitude = np.abs(stft)            
            spectrograms.append((magnitude, start_time))
    
        return spectrograms
    
    def compute_spectrogram(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path)
        return np.abs(librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length))
