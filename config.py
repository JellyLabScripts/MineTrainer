raw_training_dataset_dir = 'raw_dataset/raw_training_dataset'
raw_validation_dataset_dir = 'raw_dataset/raw_validation_dataset'
preprocessed_dataset_dir = 'preprocessed_dataset/preprocessed_training_dataset'
validation_dataset_dir = 'preprocessed_dataset/preprocessed_validation_dataset'
saved_model_path = 'saved_model/minecraft_behavior_model.keras'
lite_model_path = 'tflite_model/minecraft_behavior_model.tflite'
checkpoint_path = 'saved_model/minecraft_behavior_model.weights.h5'

# Frame dimensions
frame_width = 64
frame_height = 64
n_pixels = frame_width * frame_height

# Keypress and mouse settings
n_keys = 5                # Number of keyboard keys tracked
n_clicks = 2              # Mouse clicks (left, right)

# Mouse movement bins (discretization)
mouse_x_bins = [
    -90, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 90
]
mouse_y_bins = [
    -40, -20, -10, -4, -2, 0, 2, 4, 10, 20, 40
]
n_mouse_x = len(mouse_x_bins)
n_mouse_y = len(mouse_y_bins)

# Label dimensions
total_label_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y + 1  # +1 for dummy reward

# Data structure markers
marker_byte = 255
frame_size = n_pixels + n_keys + n_clicks + 2 + 1  # + yaw, pitch, marker

# Sequence length
n_timesteps = 60

# Actual training settings
batch_size = 1
l_rate = 0.0001
GAMMA = 0.995  # reward decay for RL setting
dimension = (64, 64)
input_shape = (n_timesteps, dimension[0], dimension[1], 3)
epochs = 1
validation_split = 0.0