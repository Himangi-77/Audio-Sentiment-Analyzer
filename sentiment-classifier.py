import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('emotion_classification-model.pkl', 'rb'))

# Define emotions
emotions = {
    'calm': 'calm',
    'happy': 'happy',
    'fearful': 'fearful',
    'disgust': 'disgust'
}

def extract_feature(file_path, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_path)
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

def classify_audio(file_path):
    try:
        feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        feature = feature.reshape(1, -1)
        emotion = model.predict(feature)[0]
        return emotions.get(emotion)
    except:
        return None

# Streamlit app
st.title('Emotion Classification from Audio')
st.write('Upload an audio file and the app will classify the emotion.')

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                audio_data, sample_rate = sf.read(uploaded_file)
                sf.write('temp.wav', audio_data.T, sample_rate, 'PCM_16')
                emotion = classify_audio('temp.wav')
                if emotion is not None:
                    st.success(f"Predicted Emotion: {emotion}")
                else:
                    st.error('Error occurred during classification.')
            except Exception as e:
                st.error(f'Error occurred during file processing: {str(e)}')