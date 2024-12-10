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

    # Initial model weights (not needed for majority-based class decision, but used for weighted prediction)
    weights = {
        'Logistic Regression': 0.2,
        'Multinomial Naive Bayes': 0.2,
        'K-Nearest Neighbors': 0.2,
        'Random Forest': 0.2,
        'Deep Learning': 0.2  # Initial weight for DL model
    }

    total_weight = sum(weights.values())
    print(f"Total Weight Before Adjustment: {total_weight:.4f}")

    dl_weight = weights['Deep Learning']
    dl_confidence_diff = abs(results['Deep Learning'][0][0] - results['Deep Learning'][0][1])
    print(f"DL Model Confidence Difference: {dl_confidence_diff}")

    # If DL model confidence difference is less than 0.25, scale its weight
    if dl_confidence_diff < 0.25:
        dl_weight = dl_confidence_diff  # Scale DL weight based on confidence
        print(f"DL Model confidence low, adjusting weight to {dl_weight}")

    # Adjust remaining weight for other models
    weight_diff = weights['Deep Learning'] - dl_weight
    remaining_weight = total_weight - weight_diff
    remaining_models = [key for key in weights if key != 'Deep Learning']
    redistributed_weight = (1 - dl_weight) / len(remaining_models)

    # Update model weights
    weights['Deep Learning'] = dl_weight
    for model in remaining_models:
        weights[model] = redistributed_weight

    total_weight = sum(weights.values())
    print(f"Total Weight After Adjustment: {total_weight:.4f}")

    # Initialize vote counters for majority voting
    votes = {
        'suicide': 0,
        'non-suicide': 0
    }

    # Initialize weighted sum for final prediction
    weighted_sum = 0

    # Process predictions for each model
    for model_name, prediction in results.items():
        if model_name == 'Deep Learning':
            prob_non_suicide = prediction[0][0]
            prob_suicide = prediction[0][1]
            print(f"Using {model_name} model -\nProbability of Non-Suicide: {prob_non_suicide}")
            print(f"Probability of Suicide: {prob_suicide}")
            # Deep Learning model: vote based on the higher probability
            if prob_non_suicide > 0.5:
                votes['non-suicide'] += 1
            else:
                votes['suicide'] += 1

            # Add weighted score for DL model
            weighted_sum += weights[model_name] * prob_non_suicide
        else:
            prediction_class = prediction[0]
            print(f"Prediction using {model_name}: {prediction_class}")
            # For ML models, directly vote based on predicted class
            if prediction_class == 'non-suicide':
                votes['non-suicide'] += 1
            else:
                votes['suicide'] += 1

            # Convert ML model prediction to score (1 = non-suicide, 0 = suicide)
            score = 1 if prediction_class == 'non-suicide' else 0
            # Add weighted score for ML models
            weighted_sum += weights[model_name] * score

    # Determine final class based on majority voting
    ensemble_class = 'non-suicide' if votes['non-suicide'] > votes['suicide'] else 'suicide'
    if ensemble_class == 'non-suicide':
        weighted_sum *= 1.25
        if weighted_sum > 1:
            weighted_sum /= 1.25
    else:
        weighted_sum *= 0.75
    
    # Calculate final prediction percentage based on weighted sum
    ensemble_prediction_score = weighted_sum / total_weight
    suicide_percentage = (1 - ensemble_prediction_score) * 100

    # Print Ensemble Result and Scores
    print(f"Majority Vote: {ensemble_class}")
    print(f"Ensemble Prediction Score (Weighted): {ensemble_prediction_score:.4f}")
    print(f"Suicide Percentage: {suicide_percentage:.2f}%")

    return {
        'individual_results': results,
        'ensemble_prediction_score': ensemble_prediction_score,
        'suicide_percentage': suicide_percentage,
        'ensemble_prediction': ensemble_class
    }