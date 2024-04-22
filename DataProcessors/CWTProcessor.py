import pywt
import numpy as np
from scipy.signal import resample
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512, wavelet='morl', num_scales=128, target_size=(128, 1050)):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.target_size = target_size

    def compute_segmented_spectrograms(self, audio_path):
        # Load audio file
        y, sr = super().normalize_audio(audio_path=audio_path)

        # Split audio into segments
        segments = super().split_audio_into_segments(y=y, sr=sr, duration=3, overlap=0.5)

        # Compute CWT scalograms from segments
        spectrograms = self._compute_scalogram_from_segments(segments, sr)

        return spectrograms

    def _compute_scalogram_from_segments(self, segments, sr):
        spectrograms = []

        # Compute CWT scalogram for each segment
        for start_time, segment in segments:
            scalogram, freqs = self.compute_cwt_scalogram_single(segment, sr)
            spectrograms.append((scalogram, start_time))

        return spectrograms

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
