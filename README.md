# Pneumonia Detection Model

This project is a Convolutional Neural Network (CNN) based model to detect pneumonia from chest X-ray images. It includes a Flask-based web application for real-time predictions.

## Features
- **CNN Model**: Trained on the Chest X-Ray Images (Pneumonia) dataset.
- **Flask Web App**: A simple UI to upload images and get predictions.
- **Preprocessing**: Includes image resizing, normalization, and data augmentation.

## Dataset
The dataset used for training is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/BravinSK/Pneumonia-Detection-Model.git

2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

3. Run the Flask app:
    ```bash
    python app.py

4. Open your browser and go to:
   ```bash
    http://127.0.0.1:5000
