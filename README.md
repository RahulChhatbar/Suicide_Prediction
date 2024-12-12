# Suicide Prediction Project

**Website**: [Suicide Prediction Website](https://rahulchhatbar.github.io/Suicide_Prediction_Website/)

This project focuses on predicting suicidal tendencies using various machine learning models and a custom deep learning model. It leverages advanced classification techniques to analyze text data and identify patterns associated with suicidal intent.

## Features
- Implementation of multiple machine learning classifiers:
  - Logistic Regression
  - Multinomial Naive Bayes (MultinomialNB)
  - K-Nearest Neighbors (KNeighborsClassifier)
  - Random Forest Classifier
- Development of a custom deep learning model combining BiLSTM (Bidirectional Long Short-Term Memory) and RNN (Recurrent Neural Network).
- Creation of an ensemble model using a weighted average for probability aggregation.
- Evaluation metrics for performance comparison.

## Dataset
The dataset used in this project is sourced from [Kaggle Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch). It consists of labeled text data indicating suicidal intent. Preprocessing steps include text cleaning, tokenization, and vectorization.

## Models and Metrics
### 1. Logistic Regression
- **Accuracy**: 0.9325
- **Precision**: 0.9452
- **Recall**: 0.9188
- **F1-Score**: 0.9318

### 2. Multinomial Naive Bayes (MultinomialNB)
- **Accuracy**: 0.9052
- **Precision**: 0.8724
- **Recall**: 0.9502
- **F1-Score**: 0.9096

### 3. K-Nearest Neighbors (KNeighborsClassifier)
- **Accuracy**: 0.8754
- **Precision**: 0.9165
- **Recall**: 0.8272
- **F1-Score**: 0.8696

### 4. Random Forest Classifier
- **Accuracy**: 0.9030
- **Precision**: 0.8965
- **Recall**: 0.9123
- **F1-Score**: 0.9043

### 5. Custom Deep Learning Model (BiLSTM + RNN)
- **Accuracy**: 0.9300
- **Precision**: 0.9031
- **Recall**: 0.9640
- **F1-Score**: 0.9326
- **Loss**: 0.2075

### 6. Ensemble Model (Weighted Average)
- This model combines the probabilities from multiple classifiers using a weighted average approach, improving overall robustness and predictive performance. Evaluation metrics will be provided upon execution.

## Results
The project achieves high accuracy and F1-scores across all models, with the custom BiLSTM + RNN model demonstrating superior performance in recall and F1-score. The ensemble model provides an additional layer of robustness by combining predictions from multiple classifiers.

## Conclusion
This project demonstrates the potential of machine learning and deep learning models in sensitive tasks such as suicide prediction. It provides a comprehensive framework for analyzing text data and highlights the importance of balancing accuracy, precision, recall, and F1-score for reliable predictions.