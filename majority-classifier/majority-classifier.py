import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Identify unique values and their respective counts in the training labels
    values, counts = np.unique(y_train, return_counts=True)
    
    # Select the value corresponding to the highest count
    majority_label = values[np.argmax(counts)]
    
    # Return a NumPy array filled with the majority label for every sample in X_test
    return np.full(shape=len(X_test), fill_value=majority_label)