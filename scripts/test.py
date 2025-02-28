import tensorflow as tf
import numpy as np
from scripts.preprocess import load_images

# Load the trained model
model = tf.keras.models.load_model('E:/Projects/MedicalImageAnalysis/models/pneumonia_cnn.h5')

# Load test data
test_normal_images, _ = load_images('../data/test/NORMAL', 0)
test_pneumonia_images, _ = load_images('../data/test/PNEUMONIA', 1)
test_images = np.array(test_normal_images + test_pneumonia_images)
test_images = np.expand_dims(test_images, axis=-1)

# Predict
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate
print("Predictions:", predicted_labels)