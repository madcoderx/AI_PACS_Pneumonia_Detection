from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

import sqlite3
import os

from predict import predict_disease

app = Flask(__name__)

@app.route('/uploads/<filename>')

def uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

UPLOAD_FOLDER = "uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route('/')

def home():

    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])

def predict():

    # Uploaded file
    file = request.files['file']

    # File path
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    # Save file
    file.save(filepath)

    # Predict disease
    result, confidence = predict_disease(filepath)

    # Save to database
    conn = sqlite3.connect('pacs.db')

    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO patients(filename, prediction) VALUES (?, ?)",
        (file.filename, result)
    )

    conn.commit()

    conn.close()

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        filename=file.filename,
        image_path='/' + filepath
    )

# Run app
if __name__ == '__main__':

    app.run(host='127.0.0.1',
            port=5000,
            debug=False)