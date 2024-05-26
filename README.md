You can access dataset from https://www.kaggle.com/datasets/alyyan/emotion-detection/data

Real-Time Emotion Detection Using Deep Learning
This project implements a real-time emotion detection system using deep learning techniques. The system is designed to recognize and classify human emotions from facial expressions captured through a webcam. The following technologies and libraries are utilized in this project:

OpenCV: For image processing and capturing video feed from the webcam.
TensorFlow: For building and training the deep learning model.
Keras: For creating the neural network layers and compiling the model.
Project Components
Dataset Preparation:

The dataset is organized into training and testing directories, each containing subdirectories for each emotion category (anger, disgust, fear, happy, sad, surprise, neutral).
The load_dataset function reads the images, converts them to grayscale, resizes them to 48x48 pixels, and labels them according to their respective emotion categories.
Data Preprocessing:

The images are reshaped and normalized to have pixel values between 0 and 1.
Emotion labels are converted into integer format and then into categorical format using one-hot encoding.
Model Architecture:

A Convolutional Neural Network (CNN) is defined using Keras, comprising the following layers:
Convolutional layers for feature extraction.
MaxPooling layers for downsampling.
Flatten layer to convert 2D matrices into a 1D vector.
Dense layers for classification.
Output layer with a softmax activation function to classify the input into one of the seven emotion categories.
Model Training:

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
The model is trained for 10 epochs with the training data, and its performance is validated using the test data.
Emotion Detection from Webcam Feed:

The system captures video from the webcam in real-time.
Haar cascade classifier is used to detect faces in the video frames.
For each detected face, the region of interest (ROI) is extracted, resized, and fed into the trained model to predict the emotion.
The predicted emotion label is displayed on the video feed, along with a rectangle around the detected face.
Model Saving and Loading:

The trained model is saved to a file named emotion_detection_model.h5.
The model can be loaded later for making predictions without retraining.
Usage
To use the emotion detection system, run the script. The webcam feed will open, and the system will start detecting and displaying emotions in real-time. Press 'q' to exit the application.

This project demonstrates the integration of image processing and deep learning to create a practical application capable of recognizing human emotions from facial expressions in real-time.
