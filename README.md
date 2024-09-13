Pneumonia Detection Using CNN
=============================

Table of Contents
----------------
- Project Overview                      
- Features                            
- Installation                            
- Usage 
  - Training Mode
  - Inference Mode
  - Inference with GUI
-  Directory Structure
-  Model Training Process

Project Overview
----------------
This project involves building a CNN model to classify chest X-rays as either showing pneumonia or normal.
The model is trained on a labeled dataset of X-ray images and can be used to predict pneumonia from new images.
Additionally, it provides a Flask-based web interface for uploading images and running inference interactively.

Features
----------
-Model: A classic CNN model:
        - 3 convolutional layers + max pooling
        - generalization enhancement with data augmentation, l2 regularizers, and dropout.
        - a train and saved model already exists in case the user does not want to experience model training
-Training: Training pipeline with logging and custom callbacks for monitoring.
-Inference: Run inference through the command line or a Flask-based web interface.
-GUI: Easy-to-use graphical interface for uploading and predicting pneumonia using chest X-rays.
-Logging: Logs model performance, accuracy, and loss, as well as inference operations.

Installation
------------
1) Clone repository
2) Install the required dependencies found in requirements.txt
3) Download the data set from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
4) Place the data in the project's data directory

Usage
--------
You can run the application in different modes: to train the model, perform inference via the command line, or launch a GUI to upload images and get predictions.

**Training Mode**

python main.py train                                                                     
This will start the model training process, log the details, and save the trained model.

**Inference Mode**

python main.py inference --image_path <path_to_image>                      
This will load the model and return the prediction (Pneumonia or Normal).

**Inference with GUI**                                                         

python main.py inference_gui                                                                               
This will automatically open a web browser where you can upload an X-ray image and get the prediction result.


**Directory Structure**
Xray Image Classification                                                    
├── src/                      # Source directory for all Python scripts                                                                                     
│   ├── app.py                # Flask web app for GUI inference                                         
│   ├── data_processing.py    # Data loading and preprocessing functions                                               
│   ├── inferece.py           # Inference logic                                               
│   ├── main.py               # Entry point for running training or inference                                                      
│   ├── model_train.py        # Model creation, training, and evaluation                                          
│   ├── utils.py              # Utility functions and logging setup                                                   
├── logs/                     # Logs directory for training/inference logs                                     
├── saved_models/             # Directory to store saved models                                 
├── templates/                # HTML templates for Flask app (index, result pages)                                                   
│   ├── index.html            # Home page for uploading images                                                        
│   └── result.html           # Page for displaying the prediction results                                          
├── uploads/                  # Directory to temporarily store uploaded images                                             
├── data/                     # Directory to store datasets                              
│   ├── train/                # Training data                                                     
│   ├── validation/           # Validation data                                        
│   └── test/                 # Test data                                                               
└── requirements.txt          # File to list project dependencies                                                          

**Model Training Process**                                                                                                                                                             
The current training parameters were selected based on achieving the highest accuracy.
I experimented with various factors such as the number of epochs, data augmentation settings, filter sizes, L2 regularization rate (0.001 or 0.002), batch size, and number of layers (2 or 3).
I also tested using a pre-trained model (VGG16) but found it too complex for this specific task.
Ultimately, I opted to stick with my custom model, both for better performance and to gain more hands-on experience in building a model from scratch.
