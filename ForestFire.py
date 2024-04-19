from Model.CNN import CNN
from DataProcessors.Utils import Utils
from DataProcessors.STFTProcessor import STFTProcessor
from Model.TestMetrics import TestMetrics

def generateAndSaveSpectrograms():
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms.
    '''
    input_fire_dir = 'Data\\Pre-processed Data\\Fire'
    mfcc_output_fire_dir = 'Data\\Spectrograms\\STFT\\Fire'
     
    Utils.process_audio_directory(spectrogram_type='STFT', 
                           input_dir=input_fire_dir, 
                            output_dir=mfcc_output_fire_dir) 
    
    input_nofire_dir = 'Data\\Pre-processed Data\\NoFire\\Environment'
    mfcc_output_nofire_dir = 'Data\\Spectrograms\\STFT\\NoFire'
     
    Utils.process_audio_directory(spectrogram_type='STFT', 
                           input_dir=input_nofire_dir, 
                            output_dir=mfcc_output_nofire_dir) 
    '''
    
def trainCNNModel():
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model
    '''
    cnn = CNN()
    # Load Spectrograms
    spectrograms, labels = Utils.load_data(input_dir='Data\\Spectrograms\\STFT')

    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, epochs=1, model_output_path='Model\\Model_STFT_1.keras')
    '''

def testModel():
    # Testing and Printing Test Metrics
    # Predict on Rainforest spectrogram
    testMetrics = TestMetrics('Model\Model_STFT_1.keras', 'STFT')
    testMetrics.PrintTestMetrics(audio_path = r'Data\Pre-processed Data\NoFire\Rainforest\0a4e7e350_28.wav',
                                 trueLabel= 'no-fire',
                                 doSegmentation=False)
    
    # Predict on Test Fire
    # Utils.convert_m4a_to_wav(r'Data\Pre-processed Data\Fire\Test\BurningFire.m4a', r'Data\Pre-processed Data\Fire\Test\BurningFire.wav')
    testMetrics.PrintTestMetrics(audio_path = r'Data\Pre-processed Data\Fire\Test\BurningFire.wav',
                                 trueLabel= 'fire',
                                 doSegmentation=True)
    
def main():
    #generateAndSaveSpectrograms()
    #trainCNNModel()
    testModel()

if __name__ == "__main__":
    main()