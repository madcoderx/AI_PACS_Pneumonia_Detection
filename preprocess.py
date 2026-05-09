import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(path):

    # Read image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixels
    img = img / 255.0

    # Reshape for CNN
    img = np.reshape(img, (128, 128, 1))

    return img