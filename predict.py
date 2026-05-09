import numpy as np

from tensorflow.keras.models import load_model

from preprocess import preprocess_image

# Load trained model
model = load_model("model.h5")

# Categories
categories = ["NORMAL", "PNEUMONIA"]

def predict_disease(image_path):

    # Preprocess image
    img = preprocess_image(image_path)

    # Convert to array
    img = np.array([img])

    # Prediction
    prediction = model.predict(img)
    
    # Confidence score
    confidence = float(prediction[0][0]) * 100

    # Result
    result = categories[
        int(prediction[0][0] > 0.5)
    ]

    return result, round(confidence, 2)