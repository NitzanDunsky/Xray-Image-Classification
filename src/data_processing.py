import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def data_processing(train_dir, validation_dir, test_dir ):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        )

    # For validation and testing, just rescale
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load training images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
    )

    # Load validation images
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    test_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle= False
    )

    return train_generator, validation_generator, test_generator
