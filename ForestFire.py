import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

FRAME_SIZE = 2048
HOP_SIZE = 512

def extractSTFT(audio):
    s_scale = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    return s_scale

def plotSpectogram(Y_log_scale, sr, hop_length, y_axis='log'):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y_log_scale, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.f')
    plt.show()

def main():
    
    wildfire_sound = 'Data/WildFire/split audio 4-20210504T123341Z-001/split audio 4/videoplayback (1)_02.wav'

    ipd.Audio(wildfire_sound)

    wildfire, sr = librosa.load(wildfire_sound)

    s_scale = extractSTFT(wildfire)

    Y_log_scale = librosa.power_to_db(np.abs(s_scale) ** 2)

    plotSpectogram(Y_log_scale=Y_log_scale, sr=sr, hop_length=HOP_SIZE, y_axis='log')

main()

