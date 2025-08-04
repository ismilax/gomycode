import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import random
import re
import json

# ==== Load Models and Assets ====

model = tf.keras.models.load_model('chatbot_dl_model.keras')

with open('chatbot_ml_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('trg_index_word.pkl', 'rb') as f:
    trg_index_word = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)

with open('intents.json') as f:
    intents = json.load(f)

# intent → responses
intent_doc = {i['intent']: i['responses'] for i in intents['intents']}

# ==== Utils ====

def extract_name(sentence):
    sentence = sentence.lower()
    tokens = sentence.split()
    if "i am" in sentence:
        return tokens[tokens.index("am") + 1].capitalize()
    elif "my name is" in sentence:
        return tokens[tokens.index("is") + 1].capitalize()
    elif "this is" in sentence:
        return tokens[tokens.index("is") + 1].capitalize()
    elif "i'm" in tokens:
        return tokens[tokens.index("i'm") + 1].capitalize()
    elif "im" in tokens:
        return tokens[tokens.index("im") + 1].capitalize()
    return "Human"

def preprocess_input(sentence):
    sentence = re.sub(r'[^a-zA-Z.?!\']', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()

def dl_response(sentence):
    cleaned = preprocess_input(sentence)
    tokens = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='pre', maxlen=model.input_shape[1])
    pred = model(padded)
    pred_class = np.argmax(pred.numpy(), axis=1)[0]
    intent = trg_index_word[pred_class]
    response = random.choice(intent_doc[intent])

    if intent in ["GreetingResponse", "CourtesyGreetingResponse", "CurrentHumanQuery", "WhoAmI"]:
        name = extract_name(sentence)
        response = response.replace("<HUMAN>", name).replace("%%HUMAN%%", name)
    return response, intent

def dl_predict_intent(sentence):
    cleaned = preprocess_input(sentence)
    tokens = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='pre', maxlen=model.input_shape[1])
    pred = model(padded)
    pred_class = np.argmax(pred.numpy(), axis=1)[0]
    return trg_index_word[pred_class]

def ml_response(sentence):
    vec = vectorizer.transform([sentence])
    pred_class = ml_model.predict(vec)[0]
    intent = id_to_label[pred_class]
    response = random.choice(intent_doc[intent])

    if intent in ["GreetingResponse", "CourtesyGreetingResponse", "CurrentHumanQuery", "WhoAmI"]:
        name = extract_name(sentence)
        response = response.replace("<HUMAN>", name).replace("%%HUMAN%%", name)
    return response, intent

def ml_predict_intent(sentence):
    vec = vectorizer.transform([sentence])
    pred_class = ml_model.predict(vec)[0]
    return id_to_label[pred_class]

# ==== Streamlit Interface ====

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🤖 Chatbot & Intent Detector")

mode = st.radio("Select Mode", ["Chat with Bot", "Detect Intent Only"])
model_choice = st.selectbox("Choose Model", ["Deep Learning", "Machine Learning"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if mode == "Chat with Bot":
    user_input = st.chat_input("Say something...")
else:
    user_input = st.text_input("Type your message to detect intent:")

if user_input:
    if mode == "Chat with Bot":
        # Chat mode
        if model_choice == "Deep Learning":
            bot_reply, intent = dl_response(user_input)
        else:
            bot_reply, intent = ml_response(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", f"{bot_reply} ({intent})"))
    else:
        # Intent mode
        if model_choice == "Deep Learning":
            intent = dl_predict_intent(user_input)
        else:
            intent = ml_predict_intent(user_input)
        st.markdown(f"**Predicted Intent:** `{intent}`")

# Show chat history
if mode == "Chat with Bot":
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)
