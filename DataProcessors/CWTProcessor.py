import librosa
import pywt
import numpy as np
from DataProcessing.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    
    def compute_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path)
        wavelet = 'morl'  # Morlet wavelet
        scales = np.arange(1, 128)  # Adjusted scales
        coeffs, freqs = pywt.cwt(y, scales, wavelet)
        scalogram = np.abs(coeffs) ** 2
        return scalogram, freqs