Sentiment Analysis Using Logistic Regression
Overview
This Python script performs sentiment analysis on a sample dataset of product reviews. It uses a Logistic Regression model to classify the reviews as positive or negative. The model is trained on text data which is vectorized using a Count Vectorizer. It then evaluates the model's performance by displaying a classification report, calculating the accuracy score, and visualizing the results with a confusion matrix.

Prerequisites
Before running the code, ensure you have the following Python libraries installed:

pandas
scikit-learn
matplotlib
seaborn
You can install these using pip:

bash
Copy code
pip install pandas scikit-learn matplotlib seaborn
Code Explanation
Import Libraries:

pandas for data manipulation.
CountVectorizer from sklearn.feature_extraction.text for text vectorization.
train_test_split from sklearn.model_selection for splitting data into training and testing sets.
LogisticRegression from sklearn.linear_model for building the classification model.
classification_report, accuracy_score, confusion_matrix for evaluating model performance.
matplotlib.pyplot and seaborn for visualizing the results.
Data:

A small sample dataset with product reviews and corresponding sentiments (positive or negative).
Text Vectorization:

The reviews are converted into numerical data using Count Vectorizer which transforms the text into a matrix of token counts.
Data Splitting:

The dataset is split into training and testing sets using train_test_split with an 80-20 split.
Training the Model:

A Logistic Regression model is trained on the vectorized text data.
Prediction and Evaluation:

The trained model makes predictions on the test data.
The model's performance is evaluated using:
Classification Report (precision, recall, F1-score).
Accuracy Score to measure the overall accuracy.
Confusion Matrix:

A confusion matrix is plotted to visualize the performance of the classifier, showing the number of true positive, true negative, false positive, and false negative predictions.
Usage
Load the dataset or modify the sample data.
Run the script.
The model will be trained, and you will see the evaluation metrics printed to the console.
A confusion matrix will be displayed showing the prediction results.
Example Output
markdown
Copy code
              precision    recall  f1-score   support

     Negative       1.00      1.00      1.00         1
     Positive       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Accuracy: 1.0
Confusion Matrix
The confusion matrix shows the actual vs. predicted labels for the test data.

