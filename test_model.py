import tensorflow as tf

from visualize_preprocessed_data import *
from train import *


class ModelWrapper:
    def __init__(self):
        self.model = None
        self.input_shape = None

        print("Loading regular TensorFlow model...")
        self.model = tf.keras.models.load_model(regular_model_path, custom_objects={
            'custom_loss': custom_loss,
            'wasd_acc': wasd_acc,
            'space_acc': space_acc,
            'Lclk_acc': Lclk_acc,
            'Rclk_acc': Rclk_acc,
            'm_x_acc': m_x_acc,
            'm_y_acc': m_y_acc,
            'crit_mse': crit_mse
        })

        self.input_shape = self.model.input_shape[1:]
        print("TensorFlow model loaded. Input shape:", self.input_shape)

    def predict(self, input_data: np.ndarray):
        if input_data.shape != self.input_shape:
            print(f"Wrong input data shape! Found {input_data.shape} but the model requires {self.input_shape}")

        input_data = np.expand_dims(input_data, axis=0)
        return self.model.predict(input_data, verbose=0)


def load_data(input_dir, output_dir, name):
    x_path = os.path.join(input_dir, f"{name}_X.npy")
    y_path = os.path.join(output_dir, f"{name}_Y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Missing .npy files")
    X = np.load(x_path)
    Y = np.load(y_path)
    print(f"Loaded data shapes -> X: {X.shape}, Y: {Y.shape}")
    return X, Y


def test_model(name, input_dir, output_dir, index=0, visualize=True, print_raw_data=False):
    model = ModelWrapper()
    X, Y = load_data(input_dir, output_dir, name)

    # Normalize input
    sample_X = X[index].astype(np.float32) / 255.0
    sample_Y = Y[index][0]

    # Running inference
    print("\nRunning inference...")
    prediction = model.predict(sample_X)
    prediction = prediction[0][0] if prediction.shape[0] == 1 and prediction.ndim == 3 else prediction[0]

    # Interpret prediction
    pred_actions = interpret_actions(prediction)

    # Interpret ground truth
    truth_actions = interpret_actions(sample_Y)

    if print_raw_data:
        print("\n=== Ground Truth (Raw) ===")
        print(sample_Y)
        print("\n=== Model Prediction (Raw) ===")
        print(prediction)

    print("\n=== Interpreted Actions Comparison ===")
    print(f"{'Action':<12} | {'Ground Truth':<13} | {'Prediction':<10}")
    print("-" * 42)

    for k in truth_actions.keys():
        gt = truth_actions[k]
        pred = pred_actions.get(k, "N/A")
        print(f"{k:<12} | {str(gt):<13} | {str(pred):<10}")

    if visualize:
        visualize_preprocessed_data(input_dir, output_dir, name, index, 1, prediction=prediction)


if __name__ == "__main__":
    # Set these manually
    name = "screendata_rohan0852_2025-05-16_15-19-53-430"
    input_dir = "preprocessed_dataset/preprocessed_validation_dataset/input"
    output_dir = "preprocessed_dataset/preprocessed_validation_dataset/output"

    for i in range(66):
        test_model(name, input_dir, output_dir, index=i, visualize=True, print_raw_data=False)