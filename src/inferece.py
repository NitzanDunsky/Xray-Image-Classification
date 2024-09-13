from keras import models, preprocessing
import numpy as np
import logging
import os


def preprocess_image(img_path):
    logging.info(f"preparing  {img_path}...")
    img = preprocessing.image.load_img(img_path, target_size=(150, 150))  # Resize to match model input size
    img_array = preprocessing.image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def run_inference(model, img_array):
    logging.info(f"running prediction...")
    predictions = model.predict(img_array)
    return predictions

def interpret_binary_result(predictions):
    print (predictions)
    if predictions[0] > 0.5:
        result = "Pneumonia"
        print("Prediction: Pneumonia")
        logging.info(f"Prediction: Pneumonia.")
    else:
        result = "Normal"
        print("Prediction: Normal")
        logging.info(f"Prediction: Normal.")
    return result

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
    result = interpret_binary_result(predictions)

    logging.info("Inference completed.")

    return result
