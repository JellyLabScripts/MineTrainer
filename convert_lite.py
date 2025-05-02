import tensorflow as tf
from tensorflow.keras.models import load_model

from config import *
from train import *

# Define a function to convert the model to TFLite
def convert_to_tflite(keras_model_path, tflite_model_path):
    # Load the model, specifying the custom loss function in custom_objects
    model = load_model(keras_model_path, custom_objects={
        'custom_loss': custom_loss,
        'wasd_acc': wasd_acc,
        'Lclk_acc': Lclk_acc,
        'm_x_acc': m_x_acc,
        'm_y_acc': m_y_acc,
        'crit_mse': crit_mse})

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to the specified path
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_model_path}")


# Paths for model conversion
keras_model_path = 'saved_model/minecraft_behavior_model.keras'  # Path to the saved model
tflite_model_path = 'tflite_model/minecraft_model.tflite'  # Path to save the TFLite model

# Make sure the tflite_model directory exists
import os

os.makedirs("tflite_model", exist_ok=True)

# Convert the model to TFLite format
convert_to_tflite(keras_model_path, tflite_model_path)