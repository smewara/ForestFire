from Model.CNN import CNN
from DataProcessors.Utils import Utils
<<<<<<< HEAD
<<<<<<< HEAD
from DataProcessors.STFTProcessor import STFTProcessor
=======
>>>>>>> 1c357e21 (Mel spectrogram modification for audio segmentation)
=======
>>>>>>> 1c357e214b6e2bf6fa9fbe761ccbf9abb83f87bc

def main():
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms.
    '''
<<<<<<< HEAD
<<<<<<< HEAD
    input_fire_dir = 'Data\\Pre-processed Data\\Fire'
    stft_output_fire_dir = 'Data\\Spectrograms\\STFT\\Fire'
     
    Utils.process_audio_directory(spectrogram_type='STFT', 
                           input_dir=input_fire_dir, 
                            output_dir=stft_output_fire_dir) 
    
    input_nofire_dir = 'Data\\Pre-processed Data\\NoFire\\Environment'
    stft_output_nofire_dir = 'Data\\Spectrograms\\STFT\\NoFire'
     
    Utils.process_audio_directory(spectrogram_type='STFT', 
=======
    input_nofire_dir = 'Pre-processed Data\\NoFire\\Environment'
    stft_output_nofire_dir = 'Spectrograms\\mel\\NoFire'
     
    utils.process_audio_directory(spectrogram_type='mel', 
>>>>>>> 1c357e21 (Mel spectrogram modification for audio segmentation)
=======
    input_nofire_dir = 'Pre-processed Data\\NoFire\\Environment'
    stft_output_nofire_dir = 'Spectrograms\\mel\\NoFire'
     
    utils.process_audio_directory(spectrogram_type='mel', 
>>>>>>> 1c357e214b6e2bf6fa9fbe761ccbf9abb83f87bc
                           input_dir=input_nofire_dir, 
                            output_dir=stft_output_nofire_dir) 
    '''
    cnn = CNN()
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model
    '''
    # Load Spectrograms
<<<<<<< HEAD
<<<<<<< HEAD
    spectrograms, labels = Utils.load_data(input_dir='Data\\Spectrograms\\STFT')
=======
    spectrograms, labels = Utils.load_data(input_dir='Spectrograms\\mel')
>>>>>>> 1c357e21 (Mel spectrogram modification for audio segmentation)
=======
    spectrograms, labels = Utils.load_data(input_dir='Spectrograms\\mel')
>>>>>>> 1c357e214b6e2bf6fa9fbe761ccbf9abb83f87bc

    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, epochs=10, model_output_path='Model\\Model_STFT_10.keras')
    
    '''
    # Predict on Rainforest spectrogram
    stft = STFTProcessor()
    spectrogram = stft.compute_spectrogram('Data\\Pre-processed Data\\NoFire\\Rainforest\\0a4e7e350_4.wav')

    model = cnn.load_model('Model\\Model_STFT_10.keras')
    predictions = cnn.predict(spectrogram=spectrogram, model=model)
    print(predictions)

if __name__ == "__main__":
    main()

