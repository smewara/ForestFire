import librosa
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class MFCCProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        
    def compute_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path)
        mfccs2 = librosa.feature.mfcc(y, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs2, sr