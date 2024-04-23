import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from Model.CNN import CNN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class TestMetrics:
    def __init__(self, model_path, spectrogram_type):
        self.model_path = model_path
        self.spectrogram_type = spectrogram_type

    def PrintTestMetrics(self, spectrograms, true_labels):
        predictions = []
        cnn = CNN(self.model_path)

        if len(true_labels) != len(spectrograms):
            raise ValueError("Length of true_labels and spectrograms must be the same")
    
        for index, (spectrogram, start_time) in enumerate(spectrograms):
            prediction = cnn.predict(spectrogram)  # Assuming cnn.predict() takes a spectrogram as input
            predictions.append(prediction)
        
        class_names = ['no-fire', 'fire']
        cm = confusion_matrix(true_labels, predictions, labels=class_names)

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, pos_label='fire', zero_division=0)  # Use zero_division=0 to handle cases where precision is undefined
        recall = recall_score(true_labels, predictions, pos_label='fire', zero_division=0) 

        # Print results
        print("Test Metrics: ")
        print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
        print(cm)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        plt.figure()
        self.plot_confusion_matrix(cm, classes=class_names, normalize=False, title='Confusion Matrix')
        plt.show()

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Set `normalize=True` for normalization by row (i.e., true class).
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion Matrix:")
        else:
            print('Confusion Matrix:')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


