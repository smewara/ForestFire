import librosa
from DataProcessing.SpectrogramProcessor import SpectrogramProcessor

class MelProcessor(SpectrogramProcessor):
        
    def compute_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path)
        mel_spectrogram = librosa.feature.melspectrogram(y, n_fft=self.n_fft, hop_length=self.hop_length)
        return mel_spectrogram, sr