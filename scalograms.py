import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display

def compute_scalogram(audio_data, sample_rate):
    wavelet = 'morl'  # Morlet wavelet
    scales = np.arange(1, 128)  # Adjusted scales
    coeffs, freqs = pywt.cwt(audio_data, scales, wavelet)
    scalogram = np.abs(coeffs) ** 2
    return scalogram, freqs

def compute_spectogram(audio_data):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    return spectrogram

# Load the second audio file
audio_file2 = 'Data\WildFire\drive-download-20210504T145639Z-001\Split Files\y2mate.com - DD Ambience  Building on Fire  Blaze Inferno Wood Cracking Collapsing Loud Stressing_01.wav'
audio_data2, sample_rate2 = librosa.load(audio_file2, sr=32000)

scalogram2, freqs2 = compute_scalogram(audio_data2, sample_rate2)
spectrogram2 = compute_spectogram(audio_data2)

# Compute MFCCs
mfccs2 = librosa.feature.mfcc(y=audio_data2, sr=sample_rate2)

plt.figure(figsize=(15, 5))

# Plot STFT Spectrogram
plt.subplot(1, 4, 1)
librosa.display.specshow(spectrogram2, sr=sample_rate2, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT')

# Plot Mel Spectrogram
plt.subplot(1, 4, 2)
mel_spectrogram = librosa.feature.melspectrogram(y=audio_data2, sr=sample_rate2, n_mels=128, hop_length=512, n_fft=2048)
librosa.display.specshow(librosa.amplitude_to_db(mel_spectrogram, ref=np.max), sr=sample_rate2, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Mel Spectrogram')

# Plot MFCC
plt.subplot(1, 4, 4)  
mfccs2 = librosa.feature.mfcc(y=audio_data2, sr=sample_rate2, n_mfcc=10, hop_length=1, n_fft=1024)
librosa.display.specshow(mfccs2, sr=sample_rate2, x_axis='time', cmap='magma')
plt.colorbar()
plt.title('MFCC')

# Plot Morlet Wavelet Scalogram
plt.subplot(1, 4, 3)
plt.imshow(scalogram2, extent=[0, len(audio_data2)/sample_rate2, freqs2[-1], freqs2[0]], aspect='auto', cmap='magma', interpolation='bilinear', norm='log')
plt.colorbar(label='Intensity')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.title('Morlet Wavelet Scalogram')

plt.tight_layout()

plt.show()
