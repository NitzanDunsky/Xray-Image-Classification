import keras.src.callbacks
from keras import layers, models
from keras import regularizers
from keras import applications
import matplotlib.pyplot as plt
import os
from datetime import datetime

from keras.src.callbacks import EarlyStopping

from utils import EpochLogger
from utils import log_model_summary
from data_processing import data_processing
import logging
import tensorflow as tf


def create_model():
    # Define data preprocessing layers
    data_augmentation = models.Sequential([
        layers.Rescaling(1./255 , input_shape=(150, 150, 3)),          # Rescale pixel values to [0, 1]
        #layers.RandomFlip('horizontal'),     # Randomly flip images horizontally
        layers.RandomRotation(0.2),          # Randomly rotate images
        layers.RandomZoom(height_factor=(-0.2,0.3), width_factor=(-0.2, 0.3), fill_mode = "nearest"), # Random zoom
        layers.RandomContrast(0.05)
    ])

    #building the model (classic CNN, 3 layers of Conv2D + MaxPooling + fully connected (with dropout)
    model = models.Sequential()

    model.add(data_augmentation)

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5)) #dropt common rule of thumb

    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid')) # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam optimizer for adaptive learning rate

    return model


def train_model(model, train_generator, validation_generator, epochs, callbacks = None):
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks = callbacks
    )
    # Return the trained model and the history of training
    return model, history

def use_vgg16():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid')) # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam optimizer for adaptive learning rate

    return model

def plot_history (history):
    # Create the logs directory if it doesn't exist
    save_dir = '../logs/training'
    os.makedirs(save_dir, exist_ok=True)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the accuracy plot with timestamp
    accuracy_plot_path = os.path.join(save_dir, f'accuracy_plot_{timestamp}.png')
    plt.savefig(accuracy_plot_path)
    plt.clf()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the loss plot with timestamp
    loss_plot_path = os.path.join(save_dir, f'loss_plot_{timestamp}.png')
    plt.savefig(loss_plot_path)
    plt.clf()

def train_save_model():
    #print(tf.__version__)
    logging.info("TensorFlow version: %s", tf.__version__)

    # Check available devices
    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)
    logging.info("Available devices: %s", devices)


    # Check if CPU is available
    cpu_available = tf.config.list_physical_devices('CPU')
    if cpu_available:
        print("CPU is available for TensorFlow operations.")
    else:
        print("CPU is not available.")

    train_dir = '../data/train'
    validation_dir = '../data/val'
    test_dir = '../data/test'

    logging.info("Loading datasets...")
    train_ds, validation_ds, test_ds = data_processing (train_dir, validation_dir, test_dir)

    logging.info("Creating the model...")
    model = create_model()

    # Log the model summary
    log_model_summary(model)
    epoch_logger = EpochLogger()  # Create an instance of the custom callback
    early_stopping = keras.src.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=5, verbose=1)
    logging.info("Starting model training...")
    model, history = train_model(model, train_ds, validation_ds, 40, callbacks=[epoch_logger,early_stopping])
    logging.info("Model training completed.")

    #model = use_vgg16()

    #history =  model.fit(
    #train_ds,
    #epochs=3,  # Number of epochs to train
    #validation_data=validation_ds)

    plot_history(history)

    # Evaluate the model on the test data
    logging.info("Evaluating the model on the test data...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Save the model
    model_path = '../saved_models/my_model.keras'
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

    # Print the results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")