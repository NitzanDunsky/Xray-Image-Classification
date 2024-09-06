import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import math


def create_model():
    #building the model (classic CNN, 3 layers of Conv2D + MaxPooling + fully connected (with dropout)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5)) #dropt common rule of thumb

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid')) # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam optimizer for adaptive learning rate

    return model


def train_model(model, train_generator, validation_generator, epochs):
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator)
    # Return the trained model and the history of training
    return model, history

def plot_history (history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()