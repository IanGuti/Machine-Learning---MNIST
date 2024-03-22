from sklearn.datasets import fetch_openml

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Function to make a confusion matrix

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

# Name the dataset to mnist and put the data in X and the right awnser in y

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X = X.reshape(-1, 28, 28)



# Set "pixels" to white or black

thresh = 127
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            if X[i, j, k] <= 127:
                X[i, j, k] = 0
            else:
                X[i, j, k] = 255

# Function to remove the rows where the sum is 0

def remove_dead_space(my_matrix, control):
    new_matrix = my_matrix
    empty_matrix = []
    for idx, num in enumerate(new_matrix):
        if num.sum() > 0:
            empty_matrix.append(num)
    empty_matrix = np.array(empty_matrix)
    diff = 28 - empty_matrix.shape[0]
    zeros = np.zeros((diff, empty_matrix.shape[1]))
 
    if control == True:
 
        new_empty_matrix = np.concatenate((empty_matrix, zeros), axis=0)
    else:
        new_empty_matrix = np.concatenate((zeros, empty_matrix), axis=0)
    new_empty_matrix =new_empty_matrix.transpose()
    return new_empty_matrix

# Call function
    
X_new = []
for item in X:
    new_item = remove_dead_space(item, True)
    X_new.append(remove_dead_space(new_item, False))
    
X_new = np.array(X_new)
X_new = X_new.reshape(-1, 784)

# Look at the documentation

print(mnist.DESCR)

# Split the data into training, test and validation with a random state to get same results

X_train_val, X_test, y_train_val, y_test = train_test_split(X_new, y, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state = 42)

# Instantiate KNN

knn = KNeighborsClassifier()

# Create Parameters

param = {
    "n_neighbors": [2, 5, 10, 15],
    "weights": ["uniform", "distance"]}

# Search for the best parameters

grid_search = GridSearchCV(knn, param, cv = 5, scoring = "accuracy", refit = "True", n_jobs = -1)

# Fit the model

grid_search.fit(X_train, y_train)

# Check the best parameters

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Show confusion matrix and accuracy score for validation data

y_pred = grid_search.predict(X_train_val)
display_confusion_matrix(y_train_val, y_pred)
acc_train_val = accuracy_score(y_train_val, y_pred)
print(acc_train_val)

# Show confusion matrix and accuracy score for test data

y_pred_test = grid_search.predict(X_test)
display_confusion_matrix(y_test, y_pred_test)
acc_test = accuracy_score(y_test, y_pred_test)
print(acc_test)

# Save knn model
joblib.dump(grid_search, "model_knn.pkl")

# Load knn model
grid_search = joblib.load("model_knn.pkl")