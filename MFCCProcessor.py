import librosa
from DataProcessing.SpectrogramProcessor import SpectrogramProcessor

class MFCCProcessor(SpectrogramProcessor):
    
    def compute_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path)
        mfccs2 = librosa.feature.mfcc(y, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs2, sr