import os
import numpy as np
import matplotlib.pyplot as plt
from DataProcessors.STFTProcessor import STFTProcessor
from DataProcessors.MelProcessor import MelProcessor
from DataProcessors.MFCCProcessor import MFCCProcessor
from DataProcessors.CWTProcessor import CWTProcessor
from moviepy.editor import AudioFileClip

class Utils:
    def get_data_processor(spectrogram_type):
        # Extract spectrogram
        if (spectrogram_type.upper() == 'STFT') :
            processor = STFTProcessor()

        elif (spectrogram_type.upper() == 'MEL') :
            processor = MelProcessor()

        elif (spectrogram_type.upper() == 'MFCC') :
            processor = MFCCProcessor()

        elif (spectrogram_type.upper() == 'CWT') :
            processor = CWTProcessor()
        
        return processor
    
    def process_audio_directory(spectrogram_type, input_dir, output_dir = None, duration_in_sec=2.5, save = False):
        audio_files = [file for file in os.listdir(input_dir) if file.endswith('.wav')]
        all_spectrograms = []

        for file in audio_files:
            input_path = os.path.join(input_dir, file)
            
            # Extract spectrogram
            processor = Utils.get_data_processor(spectrogram_type)
            spectrograms = processor.compute_segmented_spectrograms(audio_path=input_path,
                                                                    duration_in_sec=duration_in_sec)
            all_spectrograms.extend(spectrograms)
        
            if (save):
                # Save spectrogram to output directory
                output_filename = os.path.splitext(file)[0]  # Remove file extension
                processor.save_spectrogram(spectrograms, output_dir, output_filename)

        return all_spectrograms

    def load_data(input_dir):
        subdirs = [os.path.join(input_dir, subdir) for subdir in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, subdir))]

        spectrograms = []
        labels = []

        for subdir in subdirs:
            if 'NOFIRE' in subdir.upper():
                label = 0 # no-fire
            elif 'FIRE' in subdir.upper():
                label = 1 # fire

            spectrogram_files = [file for file in os.listdir(subdir) if file.endswith('.npy')]

            for file in spectrogram_files:
                spectrogram = np.load(os.path.join(subdir, file))
                spectrograms.append(spectrogram)
                labels.append(label)

        # convert lists to numpy array
        spectrograms = np.array(spectrograms)
        labels = np.array(labels)

        return spectrograms, labels 

    def convert_m4a_to_wav(input_file, output_file):
        audio = AudioFileClip(input_file)
        audio.write_audiofile(output_file)

        print(f"Conversion completed: {output_file}")