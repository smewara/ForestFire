from DataProcessors.Utils import Utils
from Model.CNN import CNN
from Model.TestMetrics import TestMetrics

def generateAndSaveSpectrograms(spectrogram_type):
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms.
    input_fire_dir = 'Data\\Pre-processed Data\\Fire'
    mfcc_output_fire_dir = f'Data\\Spectrograms\\{spectrogram_type}\\Fire'
     
    Utils.process_audio_directory(spectrogram_type=spectrogram_type, 
                            input_dir=input_fire_dir, 
                            output_dir=mfcc_output_fire_dir) 
    
    input_nofire_dir = 'Data\\Pre-processed Data\\NoFire\\Environment'
    mfcc_output_nofire_dir = f'Data\\Spectrograms\\{spectrogram_type}\\NoFire'
     
    Utils.process_audio_directory(spectrogram_type=spectrogram_type, 
                            input_dir=input_nofire_dir, 
                            output_dir=mfcc_output_nofire_dir) 
    
def trainCNNModel(spectrogram_type, no_epochs):
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model
    cnn = CNN_2D()
    # Load Spectrograms
    spectrograms, labels = Utils.load_data(input_dir=f'Data\\Spectrograms\\{spectrogram_type}')

    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, epochs=no_epochs, model_output_path=f'Model\\Model_{spectrogram_type}_{no_epochs}.keras')

def testModel(spectrogram_type, model_path):
    # Testing and Printing Test Metrics
    # Predict on Rainforest spectrogram
    testMetrics = TestMetrics(model_path=model_path, spectrogram_type=spectrogram_type)
    testMetrics.PrintTestMetrics(audio_path = r'Data\Pre-processed Data\NoFire\Rainforest\0a4e7e350_28.wav',
                                 trueLabel= 'no-fire',
                                 doSegmentation=False)
    
    # Predict on Test Fire
    Utils.convert_m4a_to_wav(r'Data\Pre-processed Data\Fire\Test\2HFire.m4a', r'Data\Pre-processed Data\Fire\Test\2HFire.wav')
    testMetrics.PrintTestMetrics(audio_path = r'Data\Pre-processed Data\Fire\Test\2HFire.wav',
                                 trueLabel= 'fire',
                                 doSegmentation=True)
    
def main():
    spectrogram_type = 'STFT'
    #generateAndSaveSpectrograms(spectrogram_type = spectrogram_type)
    #trainCNNModel(spectrogram_type=spectrogram_type, no_epochs=10)
    testModel(spectrogram_type=spectrogram_type, model_path='Model\Model_STFT_10.keras')

if __name__ == "__main__":
    main()