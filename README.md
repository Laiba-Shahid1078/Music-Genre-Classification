
# Music Genre Classification Using CNN

## Overview

This project develops an end-to-end deep learning pipeline for automatic music genre classification.

Raw audio recordings are processed using Librosa and converted into mel-spectrogram representations. A Convolutional Neural Network (CNN) is then trained to classify music into 10 different genres.

A Streamlit web application is also included for interactive audio-based genre prediction.

---

## Problem Statement

Music genre classification is the task of automatically identifying the genre of a music track from its audio signal.

The objective of this project is to build a complete machine learning workflow that:

1. Loads raw audio files.
2. Processes the audio signals.
3. Converts audio into mel-spectrograms.
4. Normalizes the extracted representations.
5. Trains a CNN model.
6. Evaluates the model using multiple classification metrics.
7. Provides an interactive Streamlit interface for predictions.

---

## Dataset

The project uses the GTZAN Music Genre Dataset.

The dataset contains 1,000 audio tracks distributed across 10 music genres, with 100 tracks per genre.

### Supported Genres

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

---

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

---

## Project Workflow

```text
Raw Audio
    ↓
Audio Loading
    ↓
Audio Preprocessing
    ↓
Mel-Spectrogram Extraction
    ↓
Resize to 128 × 128
    ↓
Normalization
    ↓
Train / Validation / Test Split
    ↓
CNN Training
    ↓
Model Evaluation
    ↓
Streamlit Application
````

---

## Audio Preprocessing

Each audio file is processed using the following configuration:

* Sample Rate: 22,050 Hz
* Duration: 30 seconds
* Mel Bands: 128
* FFT Size: 2,048
* Hop Length: 512
* Final Feature Size: 128 × 128

Audio files shorter than 30 seconds are padded, while longer files are truncated.

The resulting mel-spectrogram is normalized to the range 0–1 before being used as CNN input.

---

## Mel-Spectrogram Representation

Instead of feeding raw audio samples directly into the CNN, the audio signal is converted into a mel-spectrogram.

```text
Audio Signal
     ↓
Spectrogram
     ↓
Mel-Spectrogram
     ↓
128 × 128 Feature Representation
     ↓
CNN
```

The mel-spectrogram preserves useful time-frequency information while providing an image-like representation suitable for a 2D CNN.

---

## Final CNN Architecture

The selected final model uses the following architecture:

```text
Input: 128 × 128 × 1
        ↓
Conv2D — 32 filters
        ↓
MaxPooling2D
        ↓
Conv2D — 64 filters
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense — 128 neurons
        ↓
Dropout — 0.3
        ↓
Dense — 10 neurons
        ↓
Softmax
```

Total Parameters:

**7,393,034**

Training configuration:

* Optimizer: Adam
* Initial Learning Rate: 0.001
* Loss Function: Sparse Categorical Crossentropy
* Batch Size: 32
* Maximum Epochs: 20
* Early Stopping: Enabled
* ReduceLROnPlateau: Enabled

---

## Model Experiments

Several controlled experiments were performed to understand the effect of architecture and training changes.

| Model           | Main Change                                | Parameters | Test Accuracy | Outcome                               |
| --------------- | ------------------------------------------ | ---------: | ------------: | ------------------------------------- |
| V1 — Baseline   | Basic CNN                                  |      7.39M |       **64%** | Strong baseline with overfitting      |
| V2 — GAP        | Flatten → GlobalAveragePooling2D           |      28.4K |           32% | Underfitting                          |
| V3 — Deeper CNN | Added Conv2D(128) + Batch Normalization    |      3.31M |           10% | Unstable/poor learning                |
| V4 — Lower LR   | Learning rate 0.001 → 0.0001 + callbacks   |      7.39M |           51% | More controlled but worse performance |
| Final Model     | Baseline architecture + training callbacks |      7.39M |       **64%** | Selected for deployment               |

These experiments demonstrate that increasing or decreasing model complexity does not automatically improve generalization.

---

## Final Model Results

The selected final model achieved:

| Metric                   |        Result |
| ------------------------ | ------------: |
| Test Accuracy            |    **64.00%** |
| Test Loss                |    **1.0710** |
| Best Validation Accuracy |    **68.00%** |
| Best Validation Loss     |    **0.9745** |
| Selected Weights         |  **Epoch 15** |
| Total Parameters         | **7,393,034** |

Early stopping restored the model weights from the best validation-loss point.

---

## Classification Performance

| Genre     | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Blues     |      0.64 |   0.70 |     0.67 |
| Classical |      0.82 |   0.90 |     0.86 |
| Country   |      0.64 |   0.70 |     0.67 |
| Disco     |      0.67 |   0.40 |     0.50 |
| Hip-Hop   |      0.47 |   0.70 |     0.56 |
| Jazz      |      0.83 |   0.50 |     0.62 |
| Metal     |      0.73 |   0.80 |     0.76 |
| Pop       |      0.78 |   0.70 |     0.74 |
| Reggae    |      0.58 |   0.70 |     0.64 |
| Rock      |      0.38 |   0.30 |     0.33 |

### Key Findings

* Classical achieved the strongest class-wise performance with a recall of 90% and an F1-score of 0.86.
* Rock was the most difficult class, with a recall of 30% and an F1-score of 0.33.
* Disco and Jazz also showed comparatively weaker classification performance.
* The final model achieved 64% accuracy on the held-out test set.

---

## External Testing

Additional audio tracks outside the training dataset were tested through the Streamlit application.

| Actual Genre | Predicted Genre | Confidence |
| ------------ | --------------- | ---------: |
| Jazz         | Jazz            |        88% |
| Classical    | Classical       |        97% |
| Blues        | Reggae          |        85% |
| Rock         | Reggae          |        85% |
| Hip-Hop      | Reggae          |        73% |

External testing showed mixed generalization.

The model correctly identified the tested Jazz and Classical tracks, while some Blues, Rock, and Hip-Hop tracks were incorrectly classified as Reggae.

These external tests are separate from the official GTZAN test accuracy.

---

## Streamlit Application

A Streamlit application was developed to provide an interactive prediction interface.

### Application Workflow

```text
Upload Audio
      ↓
Audio Preprocessing
      ↓
Mel-Spectrogram
      ↓
CNN Model
      ↓
Genre Probabilities
      ↓
Prediction + Confidence
```

The application allows users to:

* Upload `.wav` or `.mp3` audio.
* Listen to the uploaded audio.
* Predict the music genre.
* View prediction confidence.
* View probabilities for all genres.
* View the top three predictions.
* Receive an "Unknown / Low Confidence" result when the highest probability is below the configured threshold.

---

## Limitations

* The model supports only the 10 genres present in the training dataset.
* The classifier is a closed-set classifier and must select one of the trained classes.
* An audio track outside the supported genres may still be assigned to one of the known genres.
* High prediction confidence does not always guarantee a correct prediction.
* External testing showed weaker generalization for some genres.
* The dataset size is relatively small for a deep learning problem.

---

## Future Improvements

* Train on a larger and more diverse music dataset.
* Apply stronger audio data augmentation.
* Explore transfer learning and pretrained audio models.
* Improve out-of-distribution / unknown-class detection.
* Experiment with more advanced CNN architectures.
* Evaluate the model on a larger external test set.
* Improve deployment and monitoring.

---

## Project Structure

```text
GitHub
│
├── README.md
├── music_genre_classification.ipynb
├── app.py
├── requirements.txt
│
└── results/
    ├── experiment_results.md
    ├── final_accuracy.png
    ├── final_loss.png
    └── final_confusion_matrix.png
     
Hugging Face
│
└── music_genre_classifier.keras
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

The application will open in the browser.

---
## Trained Model

The trained CNN model is hosted on Hugging Face:

[View / Download Model](https://huggingface.co/Laibsss/music-genre-classification)

## Author

**Laiba Shahid**

Bachelor's in Data Science

````
