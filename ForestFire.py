import numpy as np
from DataProcessors.Utils import Utils
from Model.CNN import CNN
from Model.TestMetrics import TestMetrics

def save_spectrogram(spectrogram_type, input_directory, output_directory):
    Utils.process_audio_directory(spectrogram_type=spectrogram_type, 
                                  input_dir=input_directory, 
                                  output_dir=output_directory,
                                  save=True) 
    
def generateAndSaveSpectrograms(spectrogram_type):
    # This commented block creates spectrograms. Uncomment and change spectrogram type and directory
    # to create other types of spectrograms. 
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\Fire\\Original',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire')
    
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\PitchShifted',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire')
    
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\TimeStretched',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire')
    
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\Superimposed',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire')

    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\NoFire\\Environment',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\NoFire')
    
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\NoFire\\Rainforest',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\NoFire')
    
def trainCNNModel(spectrogram_type, no_of_layers, no_epochs):
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model
    cnn = CNN()
    # Load Spectrograms
    spectrograms, labels = Utils.load_data(input_dir=f'Data\\Spectrograms\\{spectrogram_type}')

    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, no_of_layers=no_of_layers, epochs=no_epochs, model_output_path=f'Model\\Model_{no_of_layers}D_{spectrogram_type}_{no_epochs}.keras')

def testModel(spectrogram_type, model_path):
    # Define base folder for test data
    base_folder = 'Data//Pre-processed Data//Test//'

    true_labels = []
    true_labels_no_fire = []
    true_labels_fire = []
    testMetrics = TestMetrics(model_path=model_path, spectrogram_type=spectrogram_type)
    processor = Utils.get_data_processor(spectrogram_type=spectrogram_type)
    
    # Process 'no-fire' (Rainforest) spectrograms
    spectrograms = Utils.process_audio_directory(spectrogram_type=spectrogram_type,
                                                input_dir=f'{base_folder}NoFire//Rainforest',
                                                duration_in_sec=2.5,
                                                output_dir=None,  
                                                save=False)  
    true_labels.extend(['no-fire'] * len(spectrograms))
   
    # Process 'fire' (Youtube fire) spectrograms
    spectrograms_fire = Utils.process_audio_directory(spectrogram_type=spectrogram_type,
                                                      input_dir=f'{base_folder}Fire',
                                                      duration_in_sec=30,
                                                      output_dir=None,  
                                                      save=False) 
    true_labels.extend(['fire'] * len(spectrograms_fire))
    spectrograms.extend(spectrograms_fire)
    
    # Evaluate the model using test metrics
    testMetrics.PrintTestMetrics(spectrograms=spectrograms, true_labels=true_labels)

def main():
    spectrogram_type = 'STFT'
    #generateAndSaveSpectrograms(spectrogram_type=spectrogram_type)
    #trainCNNModel(spectrogram_type=spectrogram_type, no_of_layers=3, no_epochs=10)
    testModel(spectrogram_type=spectrogram_type, model_path='Model\Model_3D_STFT_10.keras')

if __name__ == "__main__":
    main()