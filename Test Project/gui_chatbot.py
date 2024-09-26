import tkinter as tk
from tkinter import *
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model

# Enable GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

lemmatizer = WordNetLemmatizer()

# Load files
with open('intents.json', 'r') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

def clean_up_sentence(sentence):
    # Tokenize and lemmatize
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    # Create bag of words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for idx, word in enumerate(words):
            if word == s:
                bag[idx] = 1
    return np.array(bag)

def predict_class(sentence):
    # Predict the class
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[idx, prob] for idx, prob in enumerate(res) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': float(r[1])})
    return return_list

def get_response(ints, intents_json, user_input):
    # Get response from intents
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                break
    else:
        # Save unrecognized input
        with open('unrecognized_inputs.txt', 'a') as f:
            f.write(f"{user_input}\n")
        result = random.choice(intents_json['intents'][-1]['responses'])
    return result

def send():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#000000", font=("Arial", 12))

        ints = predict_class(msg)
        res = get_response(ints, intents, msg)

        chat_log.insert(END, "Bot: " + res + '\n\n')
        chat_log.config(state=DISABLED)
        chat_log.yview(END)

# Create GUI window
base = tk.Tk()
base.title("AI Chatbot")
base.geometry("500x600")
base.resizable(width=FALSE, height=FALSE)

# Create chat window
chat_log = scrolledtext.ScrolledText(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap='word')
chat_log.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=chat_log.yview)
chat_log['yscrollcommand'] = scrollbar.set

# Create button to send message
send_button = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                     bd=0, bg="#0080ff", activebackground="#00bfff", fg='#ffffff',
                     command=send)

# Create entry box for user to type message
entry_box = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
entry_box.bind("<Return>", lambda event: send())

# Place all components on the screen
scrollbar.place(x=476, y=6, height=486)
chat_log.place(x=6, y=6, height=486, width=470)
entry_box.place(x=6, y=501, height=90, width=365)
send_button.place(x=376, y=501, height=90, width=120)

base.mainloop()
