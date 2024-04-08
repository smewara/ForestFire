import os
import numpy as np
import matplotlib.pyplot as plt
from DataProcessing.STFTProcessor import STFTProcessor
from DataProcessing.MelProcessor import MelProcessor
from DataProcessing.MFCCProcessor import MFCCProcessor
from DataProcessing.CWTProcessor import CWTProcessor

class Utils:
    
    def process_audio_directory(spectrogram_type, input_dir, output_dir):
        audio_files = [file for file in os.listdir(input_dir) if file.endswith('.wav') or file.endswith('.mp3')]

        for file in audio_files:
            input_path = os.path.join(input_dir, file)
            
            # Extract spectrogram
            if (spectrogram_type.upper() == 'STFT') :
                processor = STFTProcessor()

            elif (spectrogram_type.upper() == 'MEL') :
                processor = MelProcessor()

            elif (spectrogram_type.upper() == 'MFCC') :
                processor = MFCCProcessor()

            elif (spectrogram_type.upper() == 'CWT') :
                processor = CWTProcessor()
                
            spectrograms = processor.compute_spectrogram(audio_path=input_path)
            
            # Save spectrogram to output directory
            output_filename = os.path.splitext(file)[0]  # Remove file extension
            processor.save_spectrogram(spectrograms, output_dir, output_filename)

    def load_data(input_dir):
        subdirs = [os.path.join(input_dir, subdir) for subdir in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, subdir))]

        spectrograms = []
        labels = []

        for subdir in subdirs:
            if 'fire' in subdir.lower():
                label = 1 # fire
            elif 'nofire' in subdir.lower():
                label = 0 # no-fire

            spectrogram_files = [file for file in os.listdir(subdir) if file.endswith('.npy')]

            for file in spectrogram_files:
                spectrogram = np.load(os.path.join(subdir, file))
                spectrograms.append(spectrogram)
                labels.append(label)

        # convert lists to numpy array
        spectrograms = np.array(spectrograms)
        labels = np.array(labels)

        return spectrograms, labels 
