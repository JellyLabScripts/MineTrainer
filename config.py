raw_training_dataset_dir = 'raw_dataset/raw_training_dataset'
raw_validation_dataset_dir = 'raw_dataset/raw_validation_dataset'
preprocessed_dataset_dir = 'preprocessed_dataset/preprocessed_training_dataset'
validation_dataset_dir = 'preprocessed_dataset/preprocessed_validation_dataset'
checkpoint_dir = 'saved_checkpoints'
saved_model_path = 'saved_model/minecraft_behavior_model.keras'
checkpoint_path = 'saved_model/minecraft_behavior_model.weights.h5'

# Lite model conversion
regular_model_path = 'saved_model/minecraft_behavior_model.keras'
target_model_for_lite_conversion_path = regular_model_path
lite_model_path = 'tflite_model/minecraft_behavior_model.tflite'

# Frame dimensions
raw_frame_width = 160
raw_frame_height = 90
raw_n_pixels = raw_frame_width * raw_frame_height
frame_width = 160
frame_height = 90
n_pixels = frame_width * frame_height

# Keypress and mouse settings
n_keys = 5                # Number of keyboard keys tracked
n_clicks = 2              # Mouse clicks (left, right)

key_labels = ['A', 'W', 'S', 'D', 'J']
click_labels = ['L', 'R']

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
total_label_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y + 1  # +1 for marker (dummy reward)

# Data structure markers
marker_byte = 255
raw_frame_size = raw_n_pixels + n_keys + n_clicks + 2 + 1  # +1 for marker
frame_size = n_pixels + n_keys + n_clicks + 2 + 1  # +1 for marker

# Sequence length
n_timesteps = 20

# Actual training settings
batch_size = 16
l_rate = 0.0001
GAMMA = 0.995  # reward decay for RL setting
dimension = (frame_height, frame_width)
input_shape = (n_timesteps, dimension[0], dimension[1], 3)
epochs = 20
validation_split = 0.0