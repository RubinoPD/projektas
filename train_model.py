import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from the text file
data_file = "data.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, # Feature matrix.
                                                    y, # Labels.
                                                    test_size=0.2, # 20% of the data will be used for testing.
                                                    random_state=42, # Random seed for reproducibility.
                                                    shuffle=True, # Shuffle the data before splitting.
                                                    stratify=y) # Ensure class distribution in training and testing sets is proportional to the original data.

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier() # Default parameters are used.

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train) # Fit the model using training features and labels.

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test) # Predict the labels for the test set.

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy as the proportion of correct predictions.
print(f"Accuracy: {accuracy * 100:.2f}%") # Print the accuracy in percentage format.
print(confusion_matrix(y_test, y_pred)) # Print the confusion matrix for a detailed performance breakdown.

# Save the trained model to a file for future use.
with open('./model', 'wb') as f: # Open the file in write-binary mode.
    pickle.dump(rf_classifier, f) # Serialize the trained model and save it to the file.

