import numpy as np
import cv2

from config import *

# Constants
DATASET_PATH = 'replay_dataset/test.txt'
num_head_bins = 2  # yaw + pitch
marker_byte = 255
frame_size = n_pixels + n_keys + n_clicks + num_head_bins + 1

# Load data
with open(DATASET_PATH, 'rb') as f:
    data = f.read()

num_frames = len(data) // frame_size
print(f"Total frames: {num_frames}")

for frame_idx in range(num_frames):
    offset = frame_idx * frame_size
    frame_pixels = data[offset : offset + n_pixels]
    key_inputs = data[offset + n_pixels : offset + n_pixels + n_keys]
    click_inputs = data[offset + n_pixels + n_keys : offset + n_pixels + n_keys + n_clicks]
    yaw_bin = data[offset + n_pixels + n_keys + n_clicks]
    pitch_bin = data[offset + n_pixels + n_keys + n_clicks + 1]
    marker = data[offset + n_pixels + n_keys + n_clicks + num_head_bins]

    if marker != 255:
        print(f"Frame {frame_idx} missing marker!")
        continue

    # Reshape grayscale image
    frame = np.frombuffer(frame_pixels, dtype=np.uint8).reshape((frame_height, frame_width))
    frame = cv2.resize(frame, (frame_width * 3, frame_height * 3), interpolation=cv2.INTER_NEAREST)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Decode key presses
    active_keys = [label for i, label in enumerate(key_labels) if key_inputs[i] != 0]
    active_clicks = [label for i, label in enumerate(click_labels) if click_inputs[i] != 0]

    # Compose text info
    key_text = f"Keys: {' '.join(active_keys) if active_keys else 'None'}"
    click_text = f"Clicks: {' '.join(active_clicks) if active_clicks else 'None'}"
    yaw_text = f"Delta yaw: {mouse_x_bins[yaw_bin]}"
    pitch_text = f"Delta pitch: {mouse_y_bins[pitch_bin]}"

    # Draw overlay text on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 0)

    cv2.putText(frame_color, key_text, (10, 30), font, font_scale, color, thickness)
    cv2.putText(frame_color, click_text, (10, 70), font, font_scale, color, thickness)
    cv2.putText(frame_color, yaw_text, (10, 110), font, font_scale, color, thickness)
    cv2.putText(frame_color, pitch_text, (10, 150), font, font_scale, color, thickness)

    # Show the frame
    cv2.imshow('Replay with Keybinds + Head Movement', frame_color)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()