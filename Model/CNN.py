import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np

class CNN:
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = self.load_model(model_path=model_path)
        else:
            self.model = None

    def _build_model(self, input_shape):
        # Define input layer with the specified input shape
        inputs = Input(shape=input_shape)

        # Convolutional layers
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Flatten layer to prepare for fully connected layers
        x = Flatten()(x)

        # Fully connected layers (dense layers)
        x = Dense(128, activation='relu')(x)
        
        # Dropout layer to prevent overfitting
        x = Dropout(0.2)(x)
        
        # Output layer with sigmoid activation for binary classification
        outputs = Dense(1, activation='sigmoid')(x)

        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])

    def train(self, spectrograms, labels, epochs, model_output_path):
        # Shuffle data indices
        indices = np.arange(len(spectrograms))
        np.random.shuffle(indices)

        # Use shuffled indices to reorder spectrograms and labels
        shuffled_spectrograms = spectrograms[indices]
        shuffled_labels = labels[indices]

        # Split data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(shuffled_spectrograms, 
                                                            shuffled_labels, 
                                                            test_size=0.2, 
                                                            random_state=100, 
                                                            shuffle=True)

        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=0.2, 
                                                          random_state=100, 
                                                          shuffle=True)
        
        # Reshape data to add channel dimension (assuming grayscale spectrograms)
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Get input shape
        input_shape = X_train.shape[1:]

        # Build the model
        self._build_model(input_shape)  

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))

        # Evaluate the model on test data
        evaluation_results = self.model.evaluate(X_test, y_test)
        print("Evaluation Results:", evaluation_results)

        # Save the trained model
        self.model.save(model_output_path)

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)
    
    def preprocess_spectrogram(self, spectrogram):
        # Resize and reshape the spectrogram to match model input shape (1025, 130, 1)
        resized_spectrogram = spectrogram[:, :130]  # Keep only the first 130 frames
        reshaped_spectrogram = resized_spectrogram[..., np.newaxis]  # Add channel dimension
        return reshaped_spectrogram
    
    def predict(self, spectrogram):
        # Reshape the spectrogram to match model input shape
        preprocessed_spectrogram = self.preprocess_spectrogram(spectrogram)
        predictions = self.model.predict(np.expand_dims(preprocessed_spectrogram, axis=0))
        if (predictions > 0.5) :
            return 'fire'
        else: return 'no-fire'
