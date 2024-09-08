import tensorflow as tf
from keras import utils

def data_processing(train_dir, validation_dir, test_dir ):

    # Load datasets from directories
    train_dataset = utils.image_dataset_from_directory(
        train_dir,
        image_size=(150, 150),  # Resize to 150x150
        batch_size=32,
        label_mode='binary',
        shuffle= 'True'
    )

    validation_dataset = utils.image_dataset_from_directory(
        validation_dir,
        image_size=(150, 150),  # Resize to 150x150
        batch_size=32,
        label_mode='binary'
    )

    test_dataset = utils.image_dataset_from_directory(
        test_dir,
        image_size=(150, 150),  # Resize to 150x150
        batch_size=32,
        label_mode='binary'
    )

    return train_dataset, validation_dataset, test_dataset
