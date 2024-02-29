# Emotion-Classifier-Predicting-Feelings-from-Text

## Overview
This project focuses on building a model for classifying emotions in text data. The model is trained on a dataset containing text samples labeled with six emotion categories: sadness, joy, love, anger, fear, and surprise. The project includes both the training of the model and a real-time prediction interface for users to input text and receive emotion predictions.

## Dataset
The dataset used for training is provided in the `text.csv` file. Each record consists of a text sample and its corresponding emotion label.
six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5) 

## Files Description:

text.csv: Dataset containing text samples and corresponding emotion labels.
train_model.py: Script to train the emotion classification model.
predict_emotion.py: Script for real-time emotion prediction based on user input.
requirements.txt: List of Python packages required for the project.

# Usage
## Training the Model:

Run the train_model.py script to train the emotion classification model.

## Real-time Prediction:
After training, run the predict_emotion.py script to interactively predict emotions based on user input.

## Results
The trained model achieves good accuracy on the test set, as shown in the printed classification report.
