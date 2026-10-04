# Music Genre Classification Using CNN

## Overview

This project uses audio processing and deep learning to classify music tracks into 10 different genres.

The project converts raw audio signals into mel-spectrograms and uses a Convolutional Neural Network (CNN) to learn genre-specific patterns.

## Problem Statement

Music genre classification is the task of automatically identifying the genre of a music track from its audio signal.

The goal of this project is to build an end-to-end machine learning pipeline that takes an audio file as input and predicts one of the supported music genres.

## Dataset

The project uses the GTZAN Music Genre Dataset.

The dataset contains 10 music genres with 100 audio tracks per genre.

## Genres

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Librosa
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## Project Workflow

```text
Audio File
    ↓
Audio Preprocessing
    ↓
Mel-Spectrogram
    ↓
Feature Normalization
    ↓
Train / Validation / Test Split
    ↓
CNN Model
    ↓
Model Evaluation
    ↓
Streamlit Application

=============================================================
Audio Preprocessing

Each audio file was:
Loaded at a sample rate of 22,050 Hz.
Converted to mono.
Standardized to a 30-second duration.
Converted into a mel-spectrogram.
Resized to 128 × 128.
Normalized before being used as CNN input.
=============================================================
CNN Architecture

The final model uses:

Input: 128 × 128 × 1
        ↓
Conv2D (32 filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dropout (0.3)
        ↓
Dense (10, Softmax)
=============================================================
Final Model Results
Test Accuracy: 64%
Test Loss: 1.071
Best Validation Accuracy: 68%
Best Validation Loss: 0.9745
Total Parameters: 7,393,034
=============================================================
Class-wise Performance
The model performed best on Classical music and had more difficulty distinguishing Rock and Disco.
=============================================================
External Testing
Additional audio files from outside the training dataset were tested through the Streamlit application.
The external results showed mixed generalization. Jazz and Classical tracks were correctly classified, while some Blues, Rock, and Hip-Hop tracks were incorrectly classified as Reggae.

=============================================================
Streamlit Application

A Streamlit interface was developed that allows users to:
Upload an audio file
Process the audio
Generate its mel-spectrogram representation
Predict the music genre
View prediction confidence
View probabilities for the top predictions

The application also includes low-confidence handling for uncertain predictions.
=============================================================
Limitations
The model is trained on only 10 predefined genres.
The model may incorrectly classify genres that are outside the training classes.
External testing showed that high prediction confidence does not always guarantee a correct prediction.
=============================================================
Future Improvements
Use a larger and more diverse dataset.
Experiment with transfer learning and stronger CNN architectures.
Improve out-of-distribution detection.
Explore data augmentation for audio.
Improve model generalization on external music.


Author
Laiba Shahid
