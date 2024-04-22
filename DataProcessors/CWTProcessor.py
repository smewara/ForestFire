import os
import librosa
import pywt
import numpy as np
import soundfile as sf
import pyloudnorm
from scipy.signal import resample
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512, wavelet='morl', num_scales=128, target_size=(128, 1050)):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.target_size = target_size

    def normalize_audio(self, audio_path, sample_rate):
        y, sr = librosa.load(audio_path, sr=sample_rate)
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        normalized_audio = pyloudnorm.normalize.loudness(y, loudness, -23)
        return normalized_audio, sr

    def compute_cwt_scalogram_single(self, audio, sample_rate):
        scales = np.linspace(1, self.num_scales, self.num_scales, dtype=int)
        coeffs, freqs = pywt.cwt(audio, scales, self.wavelet)
        scalogram = np.abs(coeffs) ** 2

        padded_scalogram = np.zeros((self.num_scales, coeffs.shape[1]))
        padded_scalogram[:scalogram.shape[0], :scalogram.shape[1]] = scalogram

        padded_scalogram = (padded_scalogram - padded_scalogram.min()) / (padded_scalogram.max() - padded_scalogram.min())

        resampled_scalogram = resample(padded_scalogram, self.target_size[0], axis=0)
        resampled_scalogram = resample(resampled_scalogram, self.target_size[1], axis=1)

        return resampled_scalogram, freqs

    def compute_spectrogram(self, audio_path):
        audio, sr = self.normalize_audio(audio_path, sample_rate=16000)
        segments = self.split_audio_into_segments(audio, sr)
        spectrograms = []
        for segment, start_time in segments:
            scalogram, freqs = self.compute_cwt_scalogram_single(segment, sr)
            spectrograms.append((scalogram, start_time))
        return spectrograms
