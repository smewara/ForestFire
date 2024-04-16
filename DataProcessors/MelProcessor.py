import librosa
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor
import numpy as np

class MelProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__(n_fft=n_fft, hop_length=hop_length)

    def compute_spectrogram(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path)
        #The load method returns two variables, the time series (y) and the sample rate (sr), which is the number of samples per second.

        # Split audio into segments
        segments = SpectrogramProcessor.split_audio_into_segments(y=y, sr=sr, duration=3, overlap=0.5)

        # Compute Mel spectrograms from segments      
        spectrograms = self.compute_spectrogram_from_segments(segments)
        
        return spectrograms
    
def compute_spectrogram_from_segments(self, segments):
    spectrograms = []

    # Compute Mel for each segment
    for start_time, segment in segments:
        mel = librosa.feature.melspectrogram(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(mel)
        mel_spectrogram.append((magnitude, start_time))

    return spectrograms
