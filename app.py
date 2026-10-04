
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

SAMPLE_RATE = 22050
DURATION = 30

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

IMG_HEIGHT = 128
IMG_WIDTH = 128

# Initial confidence threshold
CONFIDENCE_THRESHOLD = 0.50

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

from huggingface_hub import hf_hub_download

MODEL_REPO = "Laibsss/music-genre-classification"
MODEL_FILENAME = "music_genre_classifier.keras"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    return tf.keras.models.load_model(model_path)


model = load_model()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_path):

    # Load audio
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    # Fixed 30-second length
    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        audio = audio[:target_length]


    # Create Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )


    # Convert to dB
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # Resize to model input size
    mel_resized = tf.image.resize(
        mel_db[..., np.newaxis],
        [IMG_HEIGHT, IMG_WIDTH]
    ).numpy()


    # Remove extra dimension
    mel_resized = mel_resized.squeeze()


    # Normalize
    feature_min = mel_resized.min()
    feature_max = mel_resized.max()

    if feature_max != feature_min:

        mel_resized = (
            mel_resized - feature_min
        ) / (
            feature_max - feature_min
        )

    else:

        mel_resized = np.zeros_like(
            mel_resized
        )


    # Add channel dimension
    mel_resized = mel_resized[..., np.newaxis]

    # Add batch dimension
    mel_resized = np.expand_dims(
        mel_resized,
        axis=0
    )

    return mel_resized


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🎵 Music Genre Classification")

st.write(
    "Upload a music audio file and the CNN model "
    "will predict its genre."
)

st.info(
    "The model is trained on 10 genres: "
    "Blues, Classical, Country, Disco, Hip-Hop, "
    "Jazz, Metal, Pop, Reggae, and Rock."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # Save uploaded file using its original extension
    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    temp_path = "temp_audio" + file_extension

    with open(temp_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # Audio player
    st.audio(uploaded_file)


    # Predict button
    if st.button("Predict Genre"):

        with st.spinner(
            "Analyzing audio..."
        ):

            try:

                # Extract features
                features = extract_features(
                    temp_path
                )

                # Get prediction probabilities
                probabilities = model.predict(
                    features,
                    verbose=0
                )[0]

                # Highest probability
                predicted_index = np.argmax(
                    probabilities
                )

                predicted_genre = GENRES[
                    predicted_index
                ]

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )


                # =================================================
                # LOW CONFIDENCE / UNKNOWN HANDLING
                # =================================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    st.success(
                        f"Predicted Genre: "
                        f"{predicted_genre.title()}"
                    )

                    st.write(
                        f"Confidence: "
                        f"{confidence * 100:.2f}%"
                    )

                else:

                    st.warning(
                        "Unknown / Low Confidence"
                    )

                    st.write(
                        f"The model's highest probability "
                        f"was only {confidence * 100:.2f}%."
                    )

                    st.write(
                        "This audio does not strongly match "
                        "any of the 10 genres the model was trained on."
                    )


                # =================================================
                # TOP PREDICTIONS
                # =================================================

                st.subheader(
                    "Genre Probabilities"
                )

                probability_data = pd.DataFrame({
                    "Genre": [
                        genre.title()
                        for genre in GENRES
                    ],
                    "Probability": [
                        float(p) * 100
                        for p in probabilities
                    ]
                })

                probability_data = (
                    probability_data
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )

                probability_data[
                    "Probability"
                ] = probability_data[
                    "Probability"
                ].round(2)


                # Show top 3
                st.write(
                    "Top 3 predictions:"
                )

                st.dataframe(
                    probability_data.head(3),
                    use_container_width=True
                )


                # Bar chart
                st.bar_chart(
                    probability_data.set_index(
                        "Genre"
                    )["Probability"]
                )


            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing the audio."
                )

                st.exception(e)


    # Clean temporary file
    if os.path.exists(temp_path):

        try:
            os.remove(temp_path)
        except:
            pass
