## Deep Prediction Hub

Overview

Welcome to Deep Prediction Hub, a Streamlit web application that provides two deep learning-based tasks: Sentiment Classification and Tumor Detection.

Tasks

1. Sentiment Classification
This task involves classifying the sentiment of a given text into "Positive" or "Negative". Users can input a review, and the application provides the sentiment classification using various models.

2.Tumor Detection
In Tumor Detection, users can upload an image, and the application uses a Convolutional Neural Network (CNN) model to determine if a tumor is present or not.
Getting Started

Prerequisites

    Python 3.6 or higher
    Required packages: streamlit, numpy, cv2, PIL, tensorflow
    Pre-trained models: PP.pkl, BP.pkl, DP.keras, RN.keras, LS.keras, CN.keras
    Trained IMDb word index: Ensure the IMDb word index is available for sentiment classification.

Installation

    Clone the repository: git clone https://github.com/yourusername/deep-prediction-hub.git

Usage

    Access the application by opening the provided URL after running the Streamlit app.

    Choose between "Sentiment Classification" and "Tumor Detection" tasks.

Sentiment Classification

    Enter a review in the text area.
    Select a model from the dropdown.
    Click "Submit" and then "Classify Sentiment."

Tumor Detection

    Upload an image using the file uploader.
    Click "Detect Tumor" to perform tumor detection.

Models

    Perceptron (PP.pkl): Perceptron-based sentiment classification model.
    Backpropagation (BP.pkl): Backpropagation-based sentiment classification model.
    DNN (DP.keras): Deep Neural Network sentiment classification model.
    RNN (RN.keras): Recurrent Neural Network sentiment classification model.
    LSTM (LS.keras): Long Short-Term Memory sentiment classification model.
    CNN (CN.keras): Convolutional Neural Network tumor detection model.

Contributing

Feel free to contribute by opening issues or submitting pull requests. Please follow the contribution guidelines.
License

This project is licensed under the MIT License - see the LICENSE file for details.
