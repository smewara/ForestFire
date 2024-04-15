import librosa
import IPython.display as ipd
import os
import numpy as np
from Model.CNN import CNN
from DataProcessors.Utils import Utils

def main():
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms.
    '''
    input_nofire_dir = 'Pre-processed Data\\NoFire\\Environment'
    stft_output_nofire_dir = 'Spectrograms\\mel\\NoFire'
     
    utils.process_audio_directory(spectrogram_type='mel', 
                           input_dir=input_nofire_dir, 
                            output_dir=stft_output_nofire_dir) 
    '''
    # Load Spectrograms
    spectrograms, labels = Utils.load_data(input_dir='Spectrograms\\mel')

    # Train model
    cnn = CNN()
    cnn.train(spectrograms=spectrograms, labels=labels, epochs=1, model_output_path='Model')

if __name__ == "__main__":
    main()

