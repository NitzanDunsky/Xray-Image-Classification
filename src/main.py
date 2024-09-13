import argparse
import logging
import os
from model_train import train_save_model
from inferece import load_and_infer
from app import app, open_browser
from threading import Timer


log_dir = '../logs'

def main():

    parser = argparse.ArgumentParser(description='Train a model, run inference with or without GUI.')
    parser.add_argument('mode', choices=['train', 'inference', 'inference_gui'], help='Mode to run: train, inference, or inference_gui')
    parser.add_argument('--image_path', type=str, help='Path to the image for inference (required for inference mode without gui)')

    args = parser.parse_args()
    # Train mode
    if args.mode == 'train':
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Running in training mode...")
        train_save_model()

    # Inference without GUI mode
    elif args.mode == 'inference':
        if not args.image_path:
            print("Error: --image_path is required for inference mode")
        else:
            os.makedirs(log_dir, exist_ok=True)
            logging.basicConfig(filename=os.path.join(log_dir, 'inference.log'), level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
            logging.info("Running inference without GUI...")
            result = load_and_infer(args.image_path)
            print(f"Inference result: {result}")

    elif args.mode == 'inference_gui':
        logging.basicConfig(filename=os.path.join(log_dir, 'inference_gui.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Running in GUI mode...")

        # Start a timer to open the browser after Flask has started
        Timer(1, open_browser).start()
        #app.run(debug=True)
        app.run()

if __name__ == "__main__":
    main()








