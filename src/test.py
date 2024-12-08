#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib
import warnings
warnings.filterwarnings("ignore")
import nltk
import numpy as np
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# In[3]:


# Initialize
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))
negation_words = [
    "not", "no", "never", "nothing", "nobody", "neither", "nowhere", "cannot", 
    "can't", "won't", "don't", "doesn't", "didn't", "hasn't", "haven't", 
    "hadn't", "isn't", "aren't", "wasn't", "weren't", "without", "none", 
    "naught", "naughtiness", "less"
]

best_model_lr = joblib.load('../Models/logistic_regression_model.pkl')
best_model_mnb = joblib.load('../Models/multinomial_naive_bayes_model.pkl')
best_model_knn = joblib.load('../Models/k_nearest_neighbors_model.pkl')
best_model_rf = joblib.load('../Models/random_forest_model.pkl')
vectorizer = joblib.load('../Models/vectorizer.pkl')
best_model_dl = load_model('../Models/best_dl_model_3.h5')

with open('../Models/tokenizer_dl_model_3.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

models = {
    'Logistic Regression': best_model_lr,
    'Multinomial Naive Bayes': best_model_mnb,
    'K-Nearest Neighbors': best_model_knn,
    'Random Forest': best_model_rf,
    'Deep Learning': best_model_dl
}


# In[4]:


def preprocess_and_predict(input_string, models, vectorizer):
    # Preprocess the text
    def preprocess_text(text):
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = word_tokenize(text)
        processed_words = []
        negation = False
        negation_word = ''
        for word in words:
            if word in negation_words:
                negation = True
                negation_word = word
            elif negation:
                if word.isdigit():
                    processed_words.append(negation_word)
                    processed_words.append(word)
                else:
                    processed_words.append(f'not_{word}')
                negation = False
                negation_word = ''
            else:
                processed_words.append(word)
        processed_words = [word for word in processed_words if word not in stopwords_set]
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in processed_words]
        return ' '.join(lemmatized_words)

    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Preprocess the input string
    processed_string = preprocess_text(input_string)
    
    # Transform the processed string using the vectorizer
    vectorized_input = vectorizer.transform([processed_string])
    
    # Predict using all models
    predictions = {model_name: model.predict(vectorized_input) for model_name, model in models.items() if model_name != 'Deep Learning'}
    
    # Predict using Deep Learning
    sequence_input = tokenizer.texts_to_sequences([processed_string])
    padded_input = pad_sequences(sequence_input, maxlen=150, padding='post')
    dl_predictions = best_model_dl.predict(padded_input)
    predictions['Deep Learning'] = dl_predictions
    
    return predictions


# In[8]:


def test_statement(test_string):
    print(f"Test String: {test_string}")
    results = preprocess_and_predict(test_string, models, vectorizer)
    for model_name, prediction in results.items():
        if model_name == 'Deep Learning':
            print(f"Using {model_name} model")
            print(f"Probability of Non-Suicide: {prediction[0][0]}")
            print(f"Probability of Suicide: {prediction[0][1]}")
        else:
            print(f"Prediction using {model_name}: {prediction[0]}")
    return results