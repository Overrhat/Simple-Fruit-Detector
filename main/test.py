import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------
# 1. Load the trained Keras model
# --------------------------------------------------
MODEL_PATH = "fruit_detector_model.keras"
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
# 4. Classify all images in a directory
# --------------------------------------------------
def classify_images(directory_path, true_label):
    """
    Classify images in the directory and display results for incorrect predictions
    or predictions with low confidence.
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
    correct_predictions = 0

    print("\nImages with wrong predictions or low confidence:")
    print("Filename\tTrue Label\tPredicted Label\tConfidence")

    for filename in image_files:
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {filename}. Skipping...")
            continue

        total_images += 1
        predicted_label, confidence = classify_entire_image(image)

        if predicted_label == true_label and confidence >= 0.8:
            correct_predictions += 1
        else:
            print(f"{filename}\t{true_label}\t{predicted_label}\t{confidence:.2f}")

    if total_images > 0:
        detection_percentage = (correct_predictions / total_images) * 100
        print(f"\nTotal images: {total_images}")
        print(f"Number of images detected as {true_label}: {correct_predictions}")
        print(f"Percentage of images detected as {true_label}: {detection_percentage:.2f}%")
    else:
        print("No images were processed.")

# --------------------------------------------------
# 5. Main workflow
# --------------------------------------------------
def main():
    directory_path = input("Please provide the path to the directory containing images: ")
    true_label = input("Please provide the true label for all images in the directory (e.g., banana): ")

    start_time = time.time()
    classify_images(directory_path, true_label)
    end_time = time.time()

    print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()