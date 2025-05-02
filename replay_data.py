import numpy as np
import cv2

from config import *

# Constants
DATASET_PATH = 'replay_dataset/test.txt'
num_head_bins = 2  # yaw + pitch
marker_byte = 255
frame_size = n_pixels + n_keys + n_clicks + num_head_bins + 1

# Key labels and positions
key_labels = ['A', 'W', 'S', 'D', 'Jump', 'LMB', 'RMB']
key_positions = {
    'W': (150, 30),
    'A': (110, 60),
    'S': (150, 60),
    'D': (190, 60),
    'Jump': (250, 100),
    'LMB': (30, 140),
    'RMB': (80, 140)
}

# Load data
with open(DATASET_PATH, 'rb') as f:
    data = f.read()

num_frames = len(data) // frame_size
print(f"Total frames: {num_frames}")

for frame_idx in range(num_frames):
    offset = frame_idx * frame_size
    frame_pixels = data[offset : offset + n_pixels]
    key_inputs = data[offset + n_pixels : offset + n_pixels + n_keys]
    yaw_bin = data[offset + n_pixels + n_keys]
    pitch_bin = data[offset + n_pixels + n_keys + 1]
    marker = data[offset + frame_size - 1]

    if marker != 255:
        print(f"Frame {frame_idx} missing marker!")
        continue

    # Reshape grayscale image
    frame = np.frombuffer(frame_pixels, dtype=np.uint8).reshape((frame_height, frame_width))
    frame = cv2.resize(frame, (frame_width * 3, frame_height * 3), interpolation=cv2.INTER_NEAREST)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Overlay keybinds
    for i, pressed in enumerate(key_inputs):
        label = key_labels[i]
        pos = key_positions.get(label)
        if pos:
            color = (0, 255, 0) if pressed == 1 else (100, 100, 100)
            cv2.putText(frame_color, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Overlay head movement direction
    center = (frame_color.shape[1] // 2, frame_color.shape[0] // 2)
    yaw_val = mouse_x_bins[yaw_bin] if yaw_bin < n_mouse_x else 0
    pitch_val = mouse_y_bins[pitch_bin] if pitch_bin < n_mouse_y else 0

    # Show the frame
    cv2.imshow('Replay with Keybinds + Head Movement', frame_color)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()