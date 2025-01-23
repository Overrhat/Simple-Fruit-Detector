# evaluate_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

DATASET_PATH = 'archive/original_data_set'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
MODEL_PATH = 'fruit_detector_model.keras'  # Path to your saved model

model = load_model(MODEL_PATH)

val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

