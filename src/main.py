import argparse
import tensorflow as tf
import logging
import os
from utils import EpochLogger
from utils import log_model_summary
from model import create_model, train_model, plot_history, use_vgg16
from data_processing import data_processing
from inferece import preprocess_image, run_inference, interpret_binary_result
from keras import models

log_dir = '../logs'

def train_save_model():
    print(tf.__version__)
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
    logging.info("Starting model training...")
    model, history = train_model(model, train_ds, validation_ds, 25, callbacks=epoch_logger)
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


def load_and_infer(image_path):
    if not os.path.exists('../saved_models/my_model.keras'):
        logging.error("Model not found.")
    model_path = '../saved_models/my_model.keras'

    logging.info(f"Loading the model from {model_path}...")
    model = models.load_model(model_path)

    # Preprocess the new image
    img_array = preprocess_image(image_path)

    # Run inference
    predictions = run_inference(model, img_array)

    # Interpret and print the result for binary classification
    interpret_binary_result(predictions)

    logging.info("Inference completed.")

def main():
    """Main entry point for running the script in train or inference mode."""
    parser = argparse.ArgumentParser(description='Train a model or run inference.')
    parser.add_argument('mode', choices=['train', 'inference'],
                        help='Mode to run the script in: "train" or "inference"')
    parser.add_argument('--image_path', type=str, help='Path to the image for inference (required for inference mode)')

    args = parser.parse_args()

    if args.mode == 'train':
        # Set up logging
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

        logging.info("Running in training mode...")

        #train and save model.keras
        train_save_model()

    elif args.mode == 'inference':
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'inference.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

        logging.info("Running in inference mode...")
        if not args.image_path:
            logging.error("Image path is required for inference.")
            print("Error: --image_path is required for inference mode")
        else:
            load_and_infer(args.image_path)

if __name__ == "__main__":
    main()








