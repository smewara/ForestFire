import librosa
import pywt
import numpy as np
<<<<<<< HEAD
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        
=======
from DataProcessing.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    
>>>>>>> 1c357e214b6e2bf6fa9fbe761ccbf9abb83f87bc
    def compute_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path)
        wavelet = 'morl'  # Morlet wavelet
        scales = np.arange(1, 128)  # Adjusted scales
        coeffs, freqs = pywt.cwt(y, scales, wavelet)
        scalogram = np.abs(coeffs) ** 2
        return scalogram, freqs