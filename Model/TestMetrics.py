from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from DataProcessors.CWTProcessor import CWTProcessor
from DataProcessors.MFCCProcessor import MFCCProcessor
from DataProcessors.MelProcessor import MelProcessor
from DataProcessors.STFTProcessor import STFTProcessor
from Model.CNN import CNN


class TestMetrics:
    def __init__(self, model_path, spectrogram_type):
        self.model_path = model_path
        self.spectrogram_type = spectrogram_type

    def getDataProcessor(self):
        # Extract spectrogram
        if (self.spectrogram_type.upper() == 'STFT') :
            processor = STFTProcessor()

        elif (self.spectrogram_type.upper() == 'MEL') :
            processor = MelProcessor()

        elif (self.spectrogram_type.upper() == 'MFCC') :
            processor = MFCCProcessor()

        elif (self.spectrogram_type.upper() == 'CWT') :
            processor = CWTProcessor()
        
        return processor
    
    def PrintTestMetrics(self, audio_path, trueLabel, doSegmentation = False):
        predictions = []
        cnn = CNN(self.model_path)
        processor = self.getDataProcessor()

        if doSegmentation:
            spectrograms = processor.compute_segmented_spectrograms(audio_path)
            for i, (spectrogram, start_time) in enumerate(spectrograms):
                prediction = cnn.predict(spectrogram=spectrogram)
                predictions.append(prediction)
            
            true_labels = [trueLabel] * len(predictions)
            cm = confusion_matrix(true_labels, predictions, labels=['no-fire', 'fire'])

            # Calculate accuracy, precision, and recall
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, pos_label='fire', zero_division=0)  # Use zero_division=0 to handle cases where precision is undefined
            recall = recall_score(true_labels, predictions, pos_label='fire', zero_division=0) 

            # Print results
            print("Test Results for Audio: ", audio_path)
            print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
            print(cm)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
        else:
            spectrogram = processor.compute_spectrogram(audio_path)
            prediction = cnn.predict(spectrogram=spectrogram)
            print("Test Results for Audio: ", audio_path)
            print(prediction)