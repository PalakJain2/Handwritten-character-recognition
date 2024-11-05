# Handwritten-character-recognition
# Overview:
This project is focused on building a machine learning model for recognizing handwritten characters, specifically letters from A-Z, using various classification techniques. The model takes input images of handwritten characters and classifies them into one of the 26 alphabet classes.

The project covers end-to-end steps from data preprocessing and undersampling to implementing and evaluating multiple classifiers such as Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine.

# Dataset:
The dataset contains images of handwritten characters in grayscale. Each image is represented as a flattened array of pixel values, along with a label that denotes the character class (A-Z).

Original Size: 300,000 samples
Post-Undersampling: 15,000 samples to handle class imbalance and reduce computation time

# Project Structure:
Data Preprocessing: Prepares the data by balancing classes through undersampling, stratified sampling, and random sampling.
Modeling: Multiple classifiers are applied, including:

Naive Bayes

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)


Evaluation Metrics: After training each model, the following metrics are calculated:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Prediction Plotting: Visualization of sample predictions for qualitative analysis.

# Installation
To run this project, make sure you have the following dependencies installed:

pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn

# Usage
1. Preprocessing: Run the code to preprocess the data, including undersampling to balance class distribution.

2. Training Models: Each classifier can be trained in sequence. You can add or remove classifiers based on your requirements.

3. Evaluation: The model’s performance is evaluated using various metrics and a confusion matrix to show classification results.

4. Plotting Predictions: A few sample predictions are visualized to analyze the model’s qualitative performance.

# Code Outline
Import Libraries
The necessary libraries are imported, including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.
# Data Loading
The dataset is loaded, and initial preprocessing steps are applied to handle class imbalance.
# Data Visualization
The project includes a script to plot sample images from each class to understand the data distribution and visualize the images.
# Model Training
Each classifier is initialized, trained, and evaluated. The classifiers used are:

Random Forest Classifier

Decision Tree Classifier

Naive Bayes Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)
# Evaluation
After each model is trained, it is evaluated using the following metrics:

Accuracy

Precision, Recall, and F1 Score (for each class)

Confusion Matrix

A sample of predictions is also plotted to visually assess the model's performance.
# Results
The classifiers' performance on the test set is evaluated and compared based on the defined metrics. The best-performing classifier is determined based on accuracy, precision, recall, and F1 scores.

