import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load data
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pickle', 'rb'))
classes = pickle.load(open('classes.pickle', 'rb'))
model = load_model('model.h5')

def clean_up_sentence(sentence):
    # Tokenize and lemmatize
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    # Create bag of words array
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for idx, word in enumerate(words):
            if word == s:
                bag[idx] = 1
    return np.array(bag)

def predict_class(sentence):
    # Predict the class
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[idx, r] for idx, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    # Get a random response from the predicted intent
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    else:
        return "I'm sorry, I didn't understand that."

# Chat with the bot
print("Start chatting with the bot (type 'quit' to stop)!")
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
