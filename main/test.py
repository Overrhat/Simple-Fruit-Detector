import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('fruit_detector_model.keras')  # Adjust to .keras if you saved it that way

# Define the class names
class_names = ['apples', 'banana', 'oranges']

# Load and preprocess a test image
img_path = '../resources/images/appletest.jpg'  # Replace with the path to your test image
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (128, 128)) / 255.0  # Resize and normalize
img_array = np.expand_dims(img_resized, axis=0)    # Add batch dimension

# Make predictions
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted class: {predicted_class}")
