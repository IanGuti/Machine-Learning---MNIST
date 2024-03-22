import numpy as np
import cv2
import joblib
import streamlit as st

from skimage import io

# Load vote modell

grid_search = joblib.load("model_knn.pkl")

# Streamlit site, ask to upload image

st.title("Number predictor")
new_img = st.file_uploader("Upload an image of a number", type = ["png", "jpg", "jpeg"])

# Function to put the number in the corner

def remove_dead_space(my_matrix, control):
    new_matrix = my_matrix.reshape(28, 28)
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
#     print("4",new_empty_matrix.shape)
    return new_empty_matrix




if new_img is not None:
    read_img = io.imread(new_img, cv2.IMREAD_GRAYSCALE)
    read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
    st.image(read_img)
    
    img_resized = cv2.resize(read_img, (28, 28), interpolation = cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)

# Show use the image that was uploaded

    img_resized = img_resized.reshape(-1, 784)

    thresh = 127
    for i in range(img_resized.shape[0]):
        for j in range(img_resized.shape[1]):
            if img_resized[i, j] <= thresh:
                img_resized[i, j] = 0
            else:
                img_resized[i, j] = 255
    
    X_new = []
    new_item = remove_dead_space(img_resized, True)
    X_new.append(remove_dead_space(new_item, False))
    img_resized = np.array(X_new)

    img_resized = img_resized.reshape(-1, 784)
    
    my_predict = grid_search.predict(img_resized)
    st.write("Is the image you uploaded a/an : ", my_predict)