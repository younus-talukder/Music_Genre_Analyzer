import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Music AI Studio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# MODERN UI (FULL REPLACEMENT DESIGN)
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* APP BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #0b1220, #050814);
    color: white;
}

/* HIDE STREAMLIT DEFAULT UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* TOP NAV */
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 30px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

.brand {
    font-size: 20px;
    font-weight: 800;
}

.brand span {
    color:#7c3aed;
}

/* HERO */
.hero {
    padding: 40px 10px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 800;
    line-height: 1.2;
}

.hero p {
    color:#a1a1aa;
    font-size: 16px;
    margin-top: 10px;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

/* FEATURE BOX */
.feature {
    text-align:center;
    padding:20px;
    border-radius:16px;
    background: rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    transition:0.3s;
}

.feature:hover {
    transform: translateY(-5px);
    background: rgba(255,255,255,0.06);
}

.feature h3 {
    font-size:18px;
}

.feature p {
    color:#a1a1aa;
    font-size:13px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    font-size: 16px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 12px;
    border: 1px dashed rgba(255,255,255,0.2);
}

/* RESULT */
.result {
    padding: 30px;
    text-align:center;
    border-radius: 18px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    margin-top: 20px;
}

.result h2 {
    font-size: 14px;
    opacity: 0.9;
}

.result h1 {
    font-size: 42px;
    font-weight: 800;
}

/* SECTION TITLE */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin: 25px 0 15px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Trained_model.h5"

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource()
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found!")
        return None
    return tf.keras.models.load_model(str(MODEL_PATH))

# =========================================================
# PREPROCESSING (UNCHANGED LOGIC)
# =========================================================
def load_and_preprocess_data(file_path, target_shape=(150,150)):
    data = []

    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(
        np.ceil(
            (len(audio_data) - chunk_samples)
            / (chunk_samples - overlap_samples)
        )
    ) + 1

    for i in range(num_chunks):

        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        chunk = audio_data[start:end]

        mel = librosa.power_to_db(mel)

        mel = resize(np.expand_dims(mel, axis=-1), target_shape)

        mel = resize(np.expand_dims(mel, axis=-1), target_shape)

        data.append(mel)

    return np.array(data)

# =========================================================
# PREDICTION (UNCHANGED LOGIC)
# =========================================================
def model_prediction(X_test):

    model = load_model()
    if model is None:
        return None

    y_pred = model.predict(X_test)

    predicted = np.argmax(y_pred, axis=1)

    unique, counts = np.unique(predicted, return_counts=True)

    return unique[np.argmax(counts)]

# =========================================================
# TOP BAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="brand">🎧 Music<span>AI</span> Studio</div>
    <div style="color:#a1a1aa;font-size:13px;">
        Deep Learning Genre Classification
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>AI That Understands <br> <span style="color:#7c3aed;">Your Music</span></h1>
    <p>Upload audio and let deep learning predict the genre instantly using spectrogram intelligence.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FEATURES
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
        <h3>⚡ Fast AI</h3>
        <p>Real-time prediction using CNN model</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
        <h3>🎼 10 Genres</h3>
        <p>Rock, Jazz, Pop, Metal & more</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
        <h3>🧠 Deep Learning</h3>
        <p>Mel Spectrogram based analysis</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# UPLOAD SECTION
# =========================================================
st.markdown('<div class="section-title">Upload Music</div>', unsafe_allow_html=True)

file = st.file_uploader("", type=["mp3","wav"])

if file is not None:

    audio_bytes = file.read()

    temp_path = APP_DIR / "temp_audio" / file.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(audio_bytes)

    st.audio(audio_bytes)

    col1, col2 = st.columns(2)

    with col1:
        play = st.button("▶ Play")

    with col2:
        predict = st.button("🚀 Predict Genre")

    if predict:

        with st.spinner("Analyzing audio..."):

            X = load_and_preprocess_data(str(temp_path))
            result = model_prediction(X)

            labels = [
                'blues','classical','country','disco','hiphop',
                'jazz','metal','pop','reggae','rock'
            ]

            if result is not None:

                st.markdown(f"""
                <div class="result">
                    <h2>PREDICTED GENRE</h2>
                    <h1>{labels[result].upper()}</h1>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("Prediction failed")