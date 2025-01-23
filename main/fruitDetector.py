import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------
# 1. Load the trained Keras model
# --------------------------------------------------
MODEL_PATH = "fruit_detector_model.keras"  # path to your saved Keras model
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    exit(1)

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Make sure these match the order in which your model was trained.
class_names = ["apples", "banana", "oranges"]

# Use a dictionary for your colors (BGR)
color_map = {
    "apples":  (0, 0, 255),   # Red
    "banana":  (0, 255, 255), # Yellow
    "oranges": (0, 165, 255)  # Orange
}

# --------------------------------------------------
# 2. Utility functions
# --------------------------------------------------
def prompt_for_valid_image():
    """
    Prompts the user for an image filename within the ../resources/photos/ directory.
    Returns the loaded BGR image or exits the program if user types 'exit'.
    """
    while True:
        image_filename = input("Please provide the name of an image in resources/photos (type exit to quit): ")
        if image_filename.lower() == "exit":
            print("Closing the program...")
            exit(0)
        image_path = os.path.join("..", "resources", "images", image_filename)
        loaded_image = cv2.imread(image_path)
        if loaded_image is not None:
            return loaded_image, image_filename
        else:
            print(f"Invalid image '{image_filename}'! Please try again.")

def scale_image_for_display(image, max_width=1024, max_height=768):
    """
    If image dimensions exceed max_width or max_height, resize it proportionally for display.
    """
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# --------------------------------------------------
# 3. Classify the entire image and draw a box around it
# --------------------------------------------------
def classify_entire_image(image):
    """
    Resizes the whole image to (128,128) (assuming your model expects that),
    classifies it with the CNN, and returns (label, confidence).
    """
    # Convert BGR to RGB (optional, depending on how your model was trained)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input (128x128 in your example)
    resized = cv2.resize(image_rgb, (128, 128))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (3,)
    best_idx = np.argmax(predictions)
    best_label = class_names[best_idx]
    best_confidence = float(predictions[best_idx])
    return best_label, best_confidence

def draw_border_for_label(image, label, confidence, threshold=0.80):
    """
    If confidence >= threshold, draw a border around the entire image
    in the color associated with 'label'. Also add text at the top-left corner.
    """
    if confidence < threshold:
        return image  # No bounding box if confidence too low

    color = color_map.get(label, (255, 255, 255))  # default to white if not recognized
    h, w = image.shape[:2]

    # Thickness of the border
    border_thickness = 5

    # Draw a rectangle covering the edges of the image
    # Top border
    cv2.rectangle(image, (0, 0), (w, border_thickness), color, -1)
    # Bottom border
    cv2.rectangle(image, (0, h - border_thickness), (w, h), color, -1)
    # Left border
    cv2.rectangle(image, (0, 0), (border_thickness, h), color, -1)
    # Right border
    cv2.rectangle(image, (w - border_thickness, 0), (w, h), color, -1)

    # Put text in the top-left corner (just inside the border)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return image

# --------------------------------------------------
# 4. Main workflow
# --------------------------------------------------
def main():
    input_image, image_filename = prompt_for_valid_image()
    start_time = time.time()

    # Classify the entire image
    label, conf = classify_entire_image(input_image)
    print(f"Classification: {label} with confidence {conf:.2f}")

    # Draw the border if confidence is above threshold (example: 0.80)
    draw_border_for_label(input_image, label, conf, threshold=0.80)

    # Scale for display
    output_image = scale_image_for_display(input_image)
    window_name = f"{image_filename} - Classified as {label}"
    cv2.imshow(window_name, output_image)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Window closed. Done!")

if __name__ == "__main__":
    main()
