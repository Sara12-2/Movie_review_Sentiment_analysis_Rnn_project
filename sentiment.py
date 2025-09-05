import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np

# ---- Load Trained Model ----
model = tf.keras.models.load_model("sentiment_rnn_improved.h5")

# ---- Word Index Mapping ----
word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

max_features = 10000
maxlen = 200

# ---- Encode Review Function ----
def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START> token
    for w in words:
        if w in word_index and word_index[w] < max_features:
            encoded.append(word_index[w] + 3)
        else:
            encoded.append(2)  # <UNK>
    return sequence.pad_sequences([encoded], maxlen=maxlen)

# ---- Streamlit UI ----
st.set_page_config(page_title="Movie Review Sentiment (RNN)", page_icon="🎬", layout="wide")

st.title("🎬 Sentiment Analysis (RNN)")
st.write("Enter a movie review and the model will predict if it's *Positive* or *Negative*.")

user_input = st.text_area("✍ Write your review here:")

if st.button("🔮 Predict"):
    if user_input.strip() != "":
        encoded_review = encode_review(user_input)
        prediction = model.predict(encoded_review)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "😡 Negative"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("⚠ Please enter a review before predicting.")