from data_processing import data_processing
from model import create_model, train_model, plot_history
import tensorflow as tf


def main():
    print(tf.__version__)

    # Check available devices
    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)

    # Check if CPU is available
    cpu_available = tf.config.list_physical_devices('CPU')
    if cpu_available:
        print("CPU is available for TensorFlow operations.")
    else:
        print("CPU is not available.")

    train_dir = '../data/train'
    validation_dir = '../data/val'
    test_dir = '../data/test'

    train_generator, validation_generator, test_generator = data_processing (train_dir, validation_dir, test_dir)

    model = create_model()

    model, history = train_model(model, train_generator, validation_generator, 50)

    plot_history(history)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

    # Print the results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()








