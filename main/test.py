import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------
# 1. Load the trained Keras model
# --------------------------------------------------
MODEL_PATH = "fruit_detector_model.keras"  # Adjust if your model file is named differently
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    exit(1)

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# The class names must match the order used during model training
class_names = ["apples", "banana", "oranges"]

# --------------------------------------------------
# 2. Utility function to check if file is an image
# --------------------------------------------------
def is_image_file(filename):
    """
    Checks if a file is an image based on its extension.
    You can expand this list if needed.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    _, ext = os.path.splitext(filename.lower())
    return ext in valid_extensions

# --------------------------------------------------
# 3. Classify a single image
# --------------------------------------------------
def classify_entire_image(image):
    """
    Resizes the whole image to (128,128) (assuming your model expects that),
    classifies it with the CNN, and returns (label, confidence).
    """
    # Convert BGR to RGB if your model expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model's expected input size
    resized = cv2.resize(image_rgb, (128, 128))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (3,)
    best_idx = np.argmax(predictions)
    best_label = class_names[best_idx]
    best_confidence = float(predictions[best_idx])
    
    return best_label, best_confidence

# --------------------------------------------------
# 4. Classify all images in a given directory
# --------------------------------------------------
def classify_images_in_directory(directory_path):
    """
    Go through each image in the specified directory, classify it,
    and finally print the percentage of images classified as banana.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Get a list of valid images in the directory
    image_files = [f for f in os.listdir(directory_path) if is_image_file(f)]
    if not image_files:
        print(f"No valid images found in {directory_path}")
        return
    
    total_images = 0
    banana_count = 0
    
    for filename in image_files:
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read {filename}. Skipping...")
            continue
        
        total_images += 1
        label, conf = classify_entire_image(image)
        
        # Check if predicted label is "banana"
        if label == "banana":
            banana_count += 1
    
    if total_images > 0:
        percentage_banana = (banana_count / total_images) * 100
        print(f"Total images processed: {total_images}")
        print(f"Number of images classified as banana: {banana_count}")
        print(f"Percentage classified as banana: {percentage_banana:.2f}%")
    else:
        print("No images were processed.")

# --------------------------------------------------
# 5. Main workflow
# --------------------------------------------------
def main():
    directory_path = input("Please provide the path to the directory containing images: ")
    
    start_time = time.time()
    classify_images_in_directory(directory_path)
    end_time = time.time()
    
    print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
