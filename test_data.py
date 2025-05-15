import os
import numpy as np
import tensorflow as tf

from config import *


class ModelWrapper:
    def __init__(self, lite: bool = True):
        self.lite = lite
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None

        if self.lite:
            print("Loading TFLite model...")
            self.interpreter = tf.lite.Interpreter(model_path=lite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = tuple(self.input_details[0]['shape'])
            print("TFLite model loaded. Input shape:", self.input_shape)
        else:
            print("Loading regular TensorFlow model...")
            self.model = tf.keras.models.load_model(regular_model_path)
            self.input_shape = self.model.input_shape[1:]
            print("TensorFlow model loaded. Input shape:", self.input_shape)

    def predict(self, input_data: np.ndarray):
        if self.lite:
            if input_data.shape != self.input_shape:
                input_data = np.resize(input_data, self.input_shape)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            input_data = np.expand_dims(input_data, axis=0)
            if input_data.shape[1:] != self.input_shape:
                input_data = np.resize(input_data, (1,) + self.input_shape)
            return self.model.predict(input_data, verbose=0)

# === DATA LOADING & TESTING ===
def load_data(input_dir, output_dir, name):
    x_path = os.path.join(input_dir, f"{name}_X.npy")
    y_path = os.path.join(output_dir, f"{name}_Y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Missing .npy files")
    X = np.load(x_path)
    Y = np.load(y_path)
    print(f"Loaded data shapes -> X: {X.shape}, Y: {Y.shape}")
    return X, Y


def interpret_actions(actions, mouse_x_bins, mouse_y_bins):
    # Initialize action dictionary
    action_dict = {'W': actions[0] > 0.5, 'A': actions[1] > 0.5, 'S': actions[2] > 0.5, 'D': actions[3] > 0.5,
                   'Space': actions[4] > 0.5}

    # Handle left and right clicks (indexes 5 and 6)
    left_click = actions[5] > 0.5
    right_click = actions[6] > 0.5
    action_dict['LeftClick'] = left_click
    action_dict['RightClick'] = right_click

    # Yaw (mouse X): softmax from index 7–28
    yaw_start = 7
    pitch_start = yaw_start + len(mouse_x_bins)

    # Ensure the yaw index is within the bounds of mouse_x_bins
    yaw_bin = np.argmax(actions[yaw_start:yaw_start + len(mouse_x_bins)])
    yaw_bin = min(max(yaw_bin, 0), len(mouse_x_bins) - 1)  # Prevent out-of-range error
    yaw_delta = mouse_x_bins[yaw_bin]

    # Ensure the pitch index is within the bounds of mouse_y_bins
    pitch_bin = np.argmax(actions[pitch_start:pitch_start + len(mouse_y_bins)])
    pitch_bin = min(max(pitch_bin, 0), len(mouse_y_bins) - 1)  # Prevent out-of-range error
    pitch_delta = mouse_y_bins[pitch_bin]

    # Add yaw and pitch to the action dictionary
    action_dict['Yaw'] = yaw_delta
    action_dict['Pitch'] = pitch_delta

    return action_dict


def test_model(name, input_dir, output_dir, lite=True, index=0):
    model = ModelWrapper(lite=lite)
    X, Y = load_data(input_dir, output_dir, name)

    # Normalize input
    sample_X = X[index].astype(np.float32) / 255.0
    sample_Y = Y[index][0]

    # Running inference
    print("\nRunning inference...")
    prediction = model.predict(sample_X)
    prediction = prediction[0][0] if prediction.shape[0] == 1 and prediction.ndim == 3 else prediction[0]

    # Interpret prediction
    pred_actions = interpret_actions(prediction, mouse_x_bins, mouse_y_bins)

    # Interpret ground truth
    truth_actions = interpret_actions(sample_Y, mouse_x_bins, mouse_y_bins)

    print("\n=== Ground Truth (Raw) ===")
    print(sample_Y)

    print("\n=== Ground Truth (Interpreted Actions) ===")
    for k, v in truth_actions.items():
        print(f"{k}: {v}")

    print("\n=== Model Prediction (Raw) ===")
    print(prediction)

    print("\n=== Model Prediction (Interpreted Actions) ===")
    for k, v in pred_actions.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    # Set these manually
    name = "screendata_2025-05-10_19-51-17-231(1)"
    input_dir = "preprocessed_dataset/preprocessed_training_dataset/input"
    output_dir = "preprocessed_dataset/preprocessed_training_dataset/output"
    index = 0                       # Index of the sequence to test
    lite = True

    test_model(name, input_dir, output_dir, lite=lite, index=index)