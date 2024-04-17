from Model.CNN import CNN
from DataProcessors.Utils import Utils
from DataProcessors.MFCCProcessor import MFCCProcessor

def main():
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms.
    
    input_fire_dir = 'Data\\Pre-processed Data\\Fire'
    mfcc_output_fire_dir = 'Data\\Spectrograms\\MFCC\\Fire'
     
    Utils.process_audio_directory(spectrogram_type='MFCC', 
                           input_dir=input_fire_dir, 
                            output_dir=mfcc_output_fire_dir) 
    
    input_nofire_dir = 'Data\\Pre-processed Data\\NoFire\\Environment'
    mfcc_output_nofire_dir = 'Data\\Spectrograms\\MFCC\\NoFire'
     
    Utils.process_audio_directory(spectrogram_type='MFCC', 
                           input_dir=input_nofire_dir, 
                            output_dir=mfcc_output_nofire_dir) 
    
    cnn = CNN()
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model

    # Load Spectrograms
    spectrograms, labels = Utils.load_data(input_dir='Data\\Spectrograms\\MFCC')

    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, epochs=10, model_output_path='Model\\Model_MFCC_10.keras')

    # Predict on Rainforest spectrogram
    mfcc = MFCCProcessor()
    spectrogram = mfcc.compute_spectrogram('Data\\Pre-processed Data\\NoFire\\Rainforest\\0a4e7e350_4.wav')

    model = cnn.load_model('Model\\Model_MFCC_10.keras')
    predictions = cnn.predict(spectrogram=spectrogram, model=model)
    print(predictions)

if __name__ == "__main__":
    main()