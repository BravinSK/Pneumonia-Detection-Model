from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./models/pneumonia_cnn.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = "PNEUMONIA" if np.argmax(prediction) == 1 else "NORMAL"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)