import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from DataProcessors.STFTProcessor import STFTProcessor
from DataProcessors.MelProcessor import MelProcessor
from DataProcessors.MFCCProcessor import MFCCProcessor
from DataProcessors.CWTProcessor import CWTProcessor
from moviepy.editor import AudioFileClip

from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class Utils:
    def get_data_processor(spectrogram_type = None):
        if spectrogram_type is None:
            return SpectrogramProcessor()
        
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
    
    def process_audio_directory(spectrogram_type, 
                                input_dir, 
                                output_dir = None, 
                                duration_in_sec=2.5, 
                                save = False, 
                                kind_of_augmentation = None):
        audio_files = [file for file in os.listdir(input_dir) if file.endswith('.wav')]
        all_spectrograms = []


        for file in audio_files:
            input_path = os.path.join(input_dir, file)
            
            # Extract spectrogram
            processor = Utils.get_data_processor(spectrogram_type)
            spectrograms = processor.compute_segmented_spectrograms(audio_path=input_path,
                                                                    duration_in_sec=duration_in_sec,
                                                                    kind_of_augmentation=kind_of_augmentation)
            all_spectrograms.extend(spectrograms)
        
            if (save):
                # Save spectrogram to output directory
                output_filename = os.path.splitext(file)[0]  # Remove file extension
                processor.save_spectrogram(spectrograms, output_dir, output_filename)

        return all_spectrograms

    def list_files_recursive(directory, with_augmentated_dir):
        file_list = []
        for root, dirs, files in os.walk(directory):
            if with_augmentated_dir == False:
                # Filter out directories containing 'Augmented'
                dirs[:] = [un_augmented_dir for un_augmented_dir in dirs if 'Augmented' not in un_augmented_dir]
            
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.npy'):
                    file_list.append(file_path)
        return file_list
    
    def load_spectrograms(input_dir, with_augmentated_dir = True):
        all_spectrogram_files = Utils.list_files_recursive(input_dir, with_augmentated_dir)

        spectrograms = []
        labels = []

        for file in all_spectrogram_files:
            if 'NOFIRE' in file.upper():
                label = 0 # no-fire
            elif 'FIRE' in file.upper():
                label = 1 # fire

            spectrogram = np.load(file)
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

    def superimpose_audios(input_file_fire, input_file_background, output_dir, output_file):
        audio_fire, sr1 = librosa.load(input_file_fire, sr=16000)
        audio_background, sr2 = librosa.load(input_file_background, sr=16000)

        min_length = min(len(audio_fire), len(audio_background))
        audio_fire = audio_fire[:min_length]
        audio_background = audio_background[:min_length]

        superimposed_audio = (audio_fire * 0.7) + (audio_background * 0.3)
        sf.write(os.path.join(output_dir, output_file), superimposed_audio, sr1)

