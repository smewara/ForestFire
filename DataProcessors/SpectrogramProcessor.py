import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

class SpectrogramProcessor:
    
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def normalize_audio(self, audio_path, target_loudness=-23.0):
        # Load audio file
        audio, sr = librosa.load(audio_path)

        # TODO: Normalize loudness

        return audio, sr
    
    def compute_pitch_shifting(input_dir, n_steps=4, bins_per_octave=12):
        for file in os.listdir(input_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                # Load the audio
                file_path = os.path.join(input_dir, file)
                y, sr = librosa.load(file_path)

                # Apply pitch shifting
                pitchShifted_audio = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)

                # Construct output filepath
                output_filename = os.path.splitext(file)[0] + '_shifted.wav'
                output_path = os.path.join(input_dir, output_filename)

                # Save shifted audio
                sf.write(output_path, pitchShifted_audio, sr)

    input_dir = 'Data\\Pre-processed Data\\Fire\\Train'
    compute_pitch_shifting(input_dir)
    
    def split_audio_into_segments(self, y, sr, duration=3, overlap=0.5):
        segment_samples = int(duration * sr)
        hop_length = int(segment_samples * (1 - overlap))

        # Split audio into overlapping segments
        segments = []
        start_sample = 0
        while start_sample + segment_samples < len(y):
            segment = y[start_sample:start_sample + segment_samples]
            segments.append((start_sample / sr, segment))
            start_sample += hop_length

        return segments

    def compute_spectrogram(self, audio_path):
        raise NotImplementedError("Subclasses must implement compute_spectrogram method")

    def compute_segmented_spectrograms(self, audio_path):
        raise NotImplementedError("Subclasses must implement compute_spectrogram method")

    def save_spectrogram(self, spectrograms, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)

        for index, (spectrogram, start_time) in enumerate(spectrograms):
            segment_filename = f"{filename}_{index}"
            np.save(os.path.join(output_dir, segment_filename), spectrogram)

    def plot_spectrogram(self, spectrogram, title, sr, hop_length, y_axis='log'):
        y_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        plt.figure(figsize=(25,10))
        librosa.display.specshow(y_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(title)
        plt.show()

    def plot_scalogram(self, audio, scalogram, freq, title, sr):
        plt.subplot(1, 4, 3)
        plt.imshow(scalogram, extent=[0, len(audio)/sr, freq[-1], freq[0]], aspect='auto', cmap='magma', interpolation='bilinear', norm='log')
        plt.colorbar(label='Intensity')
        plt.xlabel('Time (s)')
        plt.ylabel('Scale')
        plt.title(title)
        plt.show()

   