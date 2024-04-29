import numpy as np
from DataProcessors.Utils import Utils
from Model.CNN import CNN
from Model.TestMetrics import TestMetrics

def save_spectrogram(spectrogram_type, input_directory, output_directory, kind_of_augmentation=None):
    Utils.process_audio_directory(spectrogram_type=spectrogram_type, 
                                  input_dir=input_directory, 
                                  output_dir=output_directory,
                                  save=True,
                                  kind_of_augmentation=kind_of_augmentation) 
    
def generateAndSaveSpectrograms(spectrogram_type, apply_augmentation=False):
    # Fire 'Original + Augmented'
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\Fire\\Original',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire\\Original')
    
    #No-Fire
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\NoFire\\Environment',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\NoFire')
    
    save_spectrogram(spectrogram_type=spectrogram_type,
                     input_directory='Data\\Pre-processed Data\\Train\\NoFire\\Rainforest',
                     output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\NoFire')

    #Augmented
    if apply_augmentation:
        #Pitch-shifted
        save_spectrogram(spectrogram_type=spectrogram_type,
                        input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\PitchShifted',
                        output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire\\Augmented\\PitchShifted',
                        kind_of_augmentation='PitchShifted')
        
        #Time-streched
        save_spectrogram(spectrogram_type=spectrogram_type,
                        input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\TimeStretched',
                        output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire\\Augmented\\TimeStretched',
                        kind_of_augmentation='TimeStretched')
        
        #Superimposed
        save_spectrogram(spectrogram_type=spectrogram_type,
                        input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\Superimposed',
                        output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire\\Augmented\\Superimposed',
                        kind_of_augmentation='Superimposed')
        
        #Volume adjusted
        save_spectrogram(spectrogram_type=spectrogram_type,
                        input_directory='Data\\Pre-processed Data\\Train\\Fire\\Augmented\\Superimposed',
                        output_directory=f'Data\\Spectrograms\\{spectrogram_type}\\Fire\\Augmented\\VolumeAdjusted',
                        kind_of_augmentation='VolumeAdjusted')
   
def trainCNNModel(spectrogram_type, no_of_layers, no_epochs, apply_augmentation, model_path):
    # The below code loads the spectrograms and trains CNN model. Uncomment the block to
    # train your own model and give a name to the model
    cnn = CNN()
    # Load Spectrograms
    spectrograms, labels = Utils.load_spectrograms(input_dir=f'Data\\Spectrograms\\{spectrogram_type}', with_augmentated_dir=apply_augmentation)
    # Train model
    cnn.train(spectrograms=spectrograms, labels=labels, no_of_layers=no_of_layers, epochs=no_epochs, model_output_path=model_path)

def get_test_spectrograms_and_labels(spectrogram_type, ):
     # Define base folder for test data
    base_folder = 'Data//Pre-processed Data//Test//'
    true_labels = []
    true_labels_no_fire = []
    true_labels_fire = []
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
    return spectrograms, true_labels

def testModel(spectrogram_type, model_path):
    testMetrics = TestMetrics(model_path=model_path, spectrogram_type=spectrogram_type)
    spectrograms, true_labels = get_test_spectrograms_and_labels(spectrogram_type=spectrogram_type)
    # Evaluate the model using test metrics
    testMetrics.PrintTestMetrics(spectrograms=spectrograms, true_labels=true_labels)

def print_model_summary(model_path):
    model = CNN().load_model(model_path=model_path)
    print('{model_path}\n')
    print(model.summary())
    
def main():
    spectrogram_type = 'STFT'
    no_of_layers = 3
    no_epochs = 10
    apply_augmentation = True
    model_path = f'Model\Model_{no_of_layers}D_{spectrogram_type}_{no_epochs}_{apply_augmentation}.keras'
    #generateAndSaveSpectrograms(spectrogram_type=spectrogram_type, apply_augmentation=apply_augmentation)
    #trainCNNModel(spectrogram_type=spectrogram_type, no_of_layers=no_of_layers, no_epochs=no_epochs, apply_augmentation=apply_augmentation, model_path=model_path)
    #print_model_summary(model_path)
    testModel(spectrogram_type=spectrogram_type, model_path=model_path)

if __name__ == "__main__":
    main()