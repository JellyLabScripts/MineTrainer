import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from config import lite_model_path, target_model_for_lite_conversion_path
from train import custom_loss, wasd_acc, space_acc, Lclk_acc, Rclk_acc, m_x_acc, m_y_acc, crit_mse, input_shape


def convert_to_tflite(keras_model_path, tflite_model_path):
    """
    Robust conversion of Keras model to TensorFlow Lite format for Keras 3/TF 2.15+
    """
    print("Loading Keras model...")
    model = tf.keras.models.load_model(keras_model_path, custom_objects={
        'custom_loss': custom_loss,
        'wasd_acc': wasd_acc,
        'space_acc': space_acc,
        'Lclk_acc': Lclk_acc,
        'Rclk_acc': Rclk_acc,
        'm_x_acc': m_x_acc,
        'm_y_acc': m_y_acc,
        'crit_mse': crit_mse
    })

    # First save as Keras format (new in Keras 3)
    keras_temp_path = "saved_model/temp_model.keras"
    print(f"Saving intermediate Keras model to {keras_temp_path}...")
    model.save(keras_temp_path)

    # Convert using the saved model
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable both TFLite and TensorFlow ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # Special handling for LSTM/ConvLSTM layers
    converter._experimental_lower_tensor_list_ops = False

    try:
        print("Attempting conversion...")
        tflite_model = converter.convert()

        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully converted and saved TFLite model to {tflite_model_path}")

        return True
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        print("\nTrying alternative conversion method...")

        # Fallback: Save as SavedModel format the old way
        try:
            saved_model_dir = "saved_model/temp_savedmodel"
            print(f"Trying SavedModel format at {saved_model_dir}...")
            tf.saved_model.save(model, saved_model_dir)

            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()

            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Fallback method worked! TFLite model saved to {tflite_model_path}")
            return True
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")
            return False
        finally:
            # Clean up
            if os.path.exists(saved_model_dir):
                import shutil
                shutil.rmtree(saved_model_dir)
    finally:
        # Clean up temp file
        if os.path.exists(keras_temp_path):
            os.remove(keras_temp_path)


if __name__ == "__main__":
    # Create directory if needed
    os.makedirs(os.path.dirname(lite_model_path), exist_ok=True)

    # Convert the model
    success = convert_to_tflite(target_model_for_lite_conversion_path, lite_model_path)

    if success:
        # Verify the model
        print("\nVerifying TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=lite_model_path)
        interpreter.allocate_tensors()

        print("\nInput details:")
        print(interpreter.get_input_details())

        print("\nOutput details:")
        print(interpreter.get_output_details())

        print("\nConversion successful!")
    else:
        print("\nConversion failed")

