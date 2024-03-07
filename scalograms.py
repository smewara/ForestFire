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

# Load the first audio file
audio_file1 = '1_10101.wav'
audio_data1, sample_rate1 = librosa.load(audio_file1, sr=32000)

# Load the second audio file
audio_file2 = '00ad36516_27.wav'
audio_data2, sample_rate2 = librosa.load(audio_file2, sr=32000)

# Compute scalograms and spectrograms
scalogram1, freqs1 = compute_scalogram(audio_data1, sample_rate1)
spectrogram1 = compute_spectogram(audio_data1)

scalogram2, freqs2 = compute_scalogram(audio_data2, sample_rate2)
spectrogram2 = compute_spectogram(audio_data2)

# Plot the subplots
plt.figure(figsize=(15, 10))

# Plot scalogram for audio_file1
plt.subplot(2, 2, 1)
plt.imshow(scalogram1, extent=[0, len(audio_data1) / sample_rate1, freqs1[-1], freqs1[0]], aspect='auto', cmap='viridis', interpolation='bilinear', norm='log')
#plt.imshow(scalogram1, extent=[0, len(audio_data1) / sample_rate1, freqs1[-1], freqs1[0]], aspect='auto', cmap='viridis')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.colorbar()
plt.title('Scalogram - Fire')

# Plot spectrogram for audio_file1
plt.subplot(2, 2, 2)
librosa.display.specshow(spectrogram1, x_axis='time', y_axis='hz', sr=sample_rate1)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Fire')

# Plot scalogram for audio_file2
plt.subplot(2, 2, 3)
plt.imshow(scalogram2, extent=[0, len(audio_data2) / sample_rate2, freqs2[-1], freqs2[0]], aspect='auto', cmap='viridis', interpolation='bilinear', norm='log')
#plt.imshow(scalogram2, extent=[0, len(audio_data2) / sample_rate2, freqs2[-1], freqs2[0]], aspect='auto', cmap='viridis')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.colorbar()
plt.title('Scalogram - Rainforest background')

# Plot spectrogram for audio_file2
plt.subplot(2, 2, 4)
librosa.display.specshow(spectrogram2, x_axis='time', y_axis='hz', sr=sample_rate2)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Rainforest background')

plt.tight_layout()
plt.show()


# When it comes to building a wildfire detection model using audio data, both spectrograms and scalograms can be useful, but the choice depends on the specific requirements and characteristics of the data.

# Spectrograms are widely used for audio analysis tasks and have been employed in various applications, including environmental sound classification and event detection. They provide a time-frequency representation of the audio signal, where the x-axis represents time, the y-axis represents frequency, and the color or intensity represents the magnitude or energy of the signal at each time-frequency point.

# Scalograms, on the other hand, are based on the Continuous Wavelet Transform (CWT) and provide a time-scale representation of the signal. The y-axis represents the scale, which is inversely related to frequency, and the color or intensity represents the magnitude of the wavelet coefficients at each time-scale point.

# Here are some considerations when choosing between spectrograms and scalograms for a wildfire detection model:

#     Time-frequency resolution: Scalograms can provide better time-frequency resolution for non-stationary signals like those associated with wildfires, where the frequency content may change rapidly over time. Scalograms can capture these transient events more accurately than traditional spectrograms.
#     Noise robustness: Scalograms are generally more robust to noise and can better distinguish between noise and signal components, which can be beneficial in noisy outdoor environments where wildfires occur.
#     Feature extraction: Both spectrograms and scalograms can be used as input features for machine learning models. However, scalograms may provide more discriminative features for wildfire detection due to their ability to capture transient and non-stationary characteristics.
#     Computational complexity: Calculating scalograms can be computationally more expensive than spectrograms, especially for longer audio signals and higher scales. This may be a consideration if you have resource constraints or need to process large amounts of data in real-time.
#     Interpretability: Spectrograms are more widely understood and interpretable, which can be advantageous for model analysis and debugging.

# In general, if your wildfire detection model needs to accurately capture and distinguish transient and non-stationary characteristics of the audio data, such as crackling sounds or rapid changes in frequency c

# content, scalograms may be a better choice. However, if computational efficiency and interpretability are more important, and the audio data is relatively stationary or dominated by consistent frequency components, spectrograms may be a more suitable option.

# Ultimately, the choice between spectrograms and scalograms will depend on the specific characteristics of your audio data, the requirements of your wildfire detection model, and the trade-offs between accuracy, computational complexity, and interpretability that you're willing to make.
