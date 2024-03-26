import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pywt

FRAME_SIZE = 2048
HOP_SIZE = 512

def extractSTFT(audio):
    s_scale = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    return s_scale

def extractMelSpectrogram(audio):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)
    return mel_spectrogram

def extractMFCC(audio):
    mfccs2 = librosa.feature.mfcc(y=audio, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)
    return mfccs2

def extractScalogram(audio):
    wavelet = 'morl'  # Morlet wavelet
    scales = np.arange(1, 128)  # Adjusted scales
    coeffs, freqs = pywt.cwt(audio, scales, wavelet)
    scalogram = np.abs(coeffs) ** 2
    return scalogram, freqs

def plotSpectogram(spectrogram, title, sr, hop_length, y_axis='log'):
    y_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(y_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

def plotScalogram(audio, scalogram, freq, title, sr):
    plt.subplot(1, 4, 3)
    plt.imshow(scalogram, extent=[0, len(audio)/sr, freq[-1], freq[0]], aspect='auto', cmap='magma', interpolation='bilinear', norm='log')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')
    plt.title(title)
    plt.show()

def main():
    wildfire_sound = 'Data\WildFire\drive-download-20210504T145639Z-001\Split Files\y2mate.com - DD Ambience  Building on Fire  Blaze Inferno Wood Cracking Collapsing Loud Stressing_01.wav'

    ipd.Audio(wildfire_sound)

    wildfire, sr = librosa.load(wildfire_sound)

    stft = extractSTFT(wildfire)
    mel = extractMelSpectrogram(wildfire)
    mfcc = extractMFCC(wildfire)
    scalogram, freq = extractScalogram(wildfire)

    plotSpectogram(spectrogram=stft, title='STFT', sr=sr, hop_length=HOP_SIZE, y_axis='log')
    plotSpectogram(spectrogram=mel, title='Mel-Spectrogram', sr=sr, hop_length=HOP_SIZE, y_axis='mel')
    plotSpectogram(spectrogram=mfcc, title='MFCC', sr=sr, hop_length=HOP_SIZE, y_axis='mel')
    plotScalogram(audio=wildfire, scalogram=scalogram, freq=freq, title='CWT Scalogram', sr=sr)


main()

