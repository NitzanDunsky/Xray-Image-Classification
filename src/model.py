from keras import layers, models
from keras import regularizers
from keras import applications
import matplotlib.pyplot as plt
import os
from datetime import datetime


def create_model():
    # Define data preprocessing layers
    data_augmentation = models.Sequential([
        layers.Rescaling(1./255 , input_shape=(150, 150, 3)),          # Rescale pixel values to [0, 1]
        layers.RandomFlip('horizontal'),     # Randomly flip images horizontally
        layers.RandomRotation(0.2),          # Randomly rotate images
        layers.RandomZoom(0.2),              # Random zoom
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
