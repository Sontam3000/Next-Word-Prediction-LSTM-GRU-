import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LEN = 50   # must match training
TOP_K = 5

st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title(" Next Word Prediction App")
st.write("Choose a model and start typing — predictions update automatically!")


@st.cache_resource
def load_lstm():
    model = load_model("models/next_word_lstm.keras")
    with open("models/tokenizer_lstm.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_resource
def load_gru():
    model = load_model("models/next_word_gru.keras")
    with open("models/tokenizer_gru.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model_choice = st.selectbox(
    "Select Model",
    ["LSTM", "GRU"]
)

if model_choice == "LSTM":
    model, tokenizer = load_lstm()
else:
    model, tokenizer = load_gru()


seed_text = st.text_input(
    "Type a sentence:",
    placeholder="to be or not to"
)

def predict_next_words(text, top_k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    token_list = pad_sequences(
        [token_list],
        maxlen=MAX_SEQUENCE_LEN,
        padding="pre"
    )

    preds = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(preds)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, "")
        prob = preds[idx]
        results.append((word, prob))

    return results


if seed_text.strip() != "":
    predictions = predict_next_words(seed_text, TOP_K)

    if predictions:
        st.subheader("🔍 Predictions")
        for word, prob in predictions:
            st.write(f"**{word}** — {prob:.3f}")
    else:
        st.info("No predictions yet.")
