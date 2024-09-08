from keras import preprocessing
import numpy as np
import logging

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
        print("Prediction: Pneumonia")
        logging.info(f"Prediction: Pneumonia.")
    else:
        print("Prediction: Normal")
        logging.info(f"Prediction: Normal.")
