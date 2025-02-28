import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_images(folder, label, img_size=(150, 150)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.abspath(os.path.join(folder, filename)) # Get image path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        img = cv2.resize(img, img_size)  # Resize image
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(label)
    return images, labels

def preprocess_data(data_dir):
    normal_images, normal_labels = load_images(os.path.join(data_dir, 'NORMAL'), 0)
    pneumonia_images, pneumonia_labels = load_images(os.path.join(data_dir, 'PNEUMONIA'), 1)

    images = np.array(normal_images + pneumonia_images)
    labels = np.array(normal_labels + pneumonia_labels)

    # Reshape images for CNN input (add channel dimension)
    images = np.expand_dims(images, axis=-1)

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=2)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val