import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Preprocessing the Data
img_size = (128, 128)
batch_size = 32

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Load training and validation data
train_data = datagen.flow_from_directory(
    'archive/original_data_set',  # Replace with your dataset path
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=['apples', 'banana', 'oranges']
)

val_data = datagen.flow_from_directory(
    'archive/original_data_set',  # Replace with your dataset path
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=['apples', 'banana', 'oranges']
)

# Step 2: Defining the Model
model = Sequential([
    Input(shape=(128, 128, 3)),  # Input shape for the images
    Conv2D(32, (3, 3), activation='relu'),  # Convolutional layer
    MaxPooling2D(2, 2),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer
    MaxPooling2D(2, 2),  # Another max pooling layer
    Flatten(),  # Flatten the feature maps
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Training the Model
history = model.fit(
    train_data,                  # Training dataset
    validation_data=val_data,    # Validation dataset
    epochs=20,                   # Number of epochs
    verbose=1                    # Print progress during training
)

# Save the trained model
model.save('fruit_detector_model.keras')

# Step 4: Visualizing Training Progress
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 5: Evaluating the Model
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Optional: Save evaluation metrics to a file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Validation Loss: {val_loss}\n")
    f.write(f"Validation Accuracy: {val_accuracy}\n")
