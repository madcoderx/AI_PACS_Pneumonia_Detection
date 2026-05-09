import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Dataset arrays
data = []
labels = []

# Categories
categories = ["NORMAL", "PNEUMONIA"]

# Dataset path
dataset_path = "dataset"

# Load dataset
for category in categories:

    path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for img_name in os.listdir(path):

        try:

            img_path = os.path.join(path, img_name)

            img = preprocess_image(img_path)

            data.append(img)

            labels.append(label)

        except:
            pass

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CNN Model
model = Sequential()

# First convolution layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,1)
    )
)

# Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Second convolution
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

# Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model.h5")

print("Model trained successfully")

# Predictions
predictions = model.predict(X_test)

y_pred = (predictions > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# --------------------------
# ACCURACY GRAPH
# --------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend([
    'Training Accuracy',
    'Validation Accuracy'
])

plt.savefig('accuracy_graph.png')

plt.show()

# --------------------------
# LOSS GRAPH
# --------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend([
    'Training Loss',
    'Validation Loss'
])

plt.savefig('loss_graph.png')

plt.show()