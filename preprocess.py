import numpy as np
import os
from tqdm import tqdm

from config import *

class ScreendataPreprocessor:
    def __init__(self, path, output_dir):
        self.path = path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def one_hot(self, index, length):
        vec = np.zeros(length, dtype=np.float32)
        if 0 <= index < length:
            vec[index] = 1.0
        return vec

    def process(self, name):
        with open(self.path, 'rb') as f:
            data = f.read()

        total_frames = len(data) // raw_frame_size
        print(f"Loaded {total_frames} frames.")

        X_seq = []
        Y_seq = []

        key_counts = np.zeros(n_keys, dtype=np.int64)
        click_counts = np.zeros(n_clicks, dtype=np.int64)
        yaw_bin_counts = np.zeros(n_mouse_x, dtype=np.int64)
        pitch_bin_counts = np.zeros(n_mouse_y, dtype=np.int64)
        total_valid_frames = 0

        for i in tqdm(range(total_frames - n_timesteps)):
            frames = []
            labels = []

            for t in range(n_timesteps):
                offset = (i + t) * raw_frame_size

                frame_data = data[offset:offset + raw_n_pixels]
                key_data = data[offset + raw_n_pixels : offset + raw_n_pixels + n_keys]
                click_data = data[offset + raw_n_pixels + n_keys : offset + raw_n_pixels + n_keys + n_clicks]
                yaw_bin = data[offset + raw_n_pixels + n_keys + n_clicks]
                pitch_bin = data[offset + raw_n_pixels + n_keys + n_clicks + 1]
                marker = data[offset + raw_frame_size - 1]

                if marker != marker_byte:
                    print(f"Frame {i+t} has invalid marker. Skipping sequence.")
                    print(f"Expected marker: {marker_byte}, Found: {marker} at frame {i + t}")
                    break

                if all(b == 0 for b in frame_data):
                    print(f"Warning: Frame {i + t} has all zero pixel data.")
                    break

                gray = np.frombuffer(frame_data, dtype=np.uint8).reshape((raw_frame_height, raw_frame_width))
                gray_downscaled = gray
                # gray_downscaled = downscale_java_style(gray, frame_height, frame_width)
                rgb = np.stack([gray_downscaled] * 3, axis=-1)

                key_array = np.frombuffer(key_data, dtype=np.uint8)
                click_array = np.frombuffer(click_data, dtype=np.uint8)

                key_counts += key_array
                click_counts += click_array
                yaw_bin_counts[yaw_bin] += 1
                pitch_bin_counts[pitch_bin] += 1
                total_valid_frames += 1

                frames.append(rgb)

                label = np.concatenate([
                    np.frombuffer(key_data, dtype=np.uint8).astype(np.float32),
                    np.frombuffer(click_data, dtype=np.uint8).astype(np.float32),
                    self.one_hot(yaw_bin, n_mouse_x),
                    self.one_hot(pitch_bin, n_mouse_y),
                    np.array([0.0], dtype=np.float32)
                ])
                labels.append(label)

            if len(frames) == n_timesteps:
                X_seq.append(frames)
                Y_seq.append(labels)

        X = np.array(X_seq, dtype=np.uint8)
        Y = np.array(Y_seq, dtype=np.float32)

        print(f"Saving {X.shape[0]} sequences of shape {X.shape[1:]}")
        input_dir = os.path.join(self.output_dir, "input")
        output_dir = os.path.join(self.output_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(input_dir, f"{name}_X.npy"), X)
        np.save(os.path.join(output_dir, f"{name}_Y.npy"), Y)

        return {
            'key_counts': key_counts,
            'click_counts': click_counts,
            'yaw_bin_counts': yaw_bin_counts,
            'pitch_bin_counts': pitch_bin_counts,
            'total_valid_frames': total_valid_frames
        }



def downscale_java_style(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Replicate one pass of Java’s downscaler:
      - average pixels in each block (integer division)
      - optionally flip vertically (Java always does)
      - integer‐only arithmetic

    Args:
        image: 2D gray np.uint8 array (h, w).
        target_height: must divide image height exactly.
        target_width: must divide image width exactly.
    Returns:
        2D np.uint8 array of shape (target_height, target_width).
    """
    height, width = image.shape
    scale_x = width // target_width
    scale_y = height // target_height

    down = np.zeros((target_height, target_width), dtype=np.uint8)

    for ty in range(target_height):
        for tx in range(target_width):
            total = 0
            count = 0
            for sy in range(scale_y):
                for sx in range(scale_x):
                    src_x = tx * scale_x + sx
                    src_y = ty * scale_y + sy

                    total += int(image[src_y, src_x])
                    count += 1

            down[ty, tx] = (total // count) if count else 0

    return down



if __name__ == "__main__":
    # Initialize statistics
    global_key_counts = np.zeros(n_keys, dtype=np.int64)
    global_click_counts = np.zeros(n_clicks, dtype=np.int64)
    global_yaw_counts = np.zeros(len(mouse_x_bins), dtype=np.int64)
    global_pitch_counts = np.zeros(len(mouse_y_bins), dtype=np.int64)
    global_total_frames = 0

    # Process training data
    txt_files = [f for f in os.listdir(raw_training_dataset_dir) if f.endswith(".txt")]
    for filename in txt_files:
        filepath = os.path.join(raw_training_dataset_dir, filename)
        name = os.path.splitext(filename)[0]
        print(f"Processing training file {filename}...")
        processor = ScreendataPreprocessor(filepath, preprocessed_dataset_dir)
        stats = processor.process(name=name)

        # Accumulate stats
        global_key_counts += stats['key_counts']
        global_click_counts += stats['click_counts']
        global_yaw_counts += stats['yaw_bin_counts']
        global_pitch_counts += stats['pitch_bin_counts']
        global_total_frames += stats['total_valid_frames']

    # Process validation data
    val_txt_files = [f for f in os.listdir(raw_validation_dataset_dir) if f.endswith(".txt")]
    for filename in val_txt_files:
        filepath = os.path.join(raw_validation_dataset_dir, filename)
        name = os.path.splitext(filename)[0]
        print(f"Processing validation file {filename}...")
        processor = ScreendataPreprocessor(filepath, validation_dataset_dir)
        stats = processor.process(name=name)

        # Accumulate stats (optional - you might want separate validation stats)
        global_key_counts += stats['key_counts']
        global_key_counts += stats['key_counts']
        global_click_counts += stats['click_counts']
        global_yaw_counts += stats['yaw_bin_counts']
        global_pitch_counts += stats['pitch_bin_counts']
        global_total_frames += stats['total_valid_frames']

    # Make sure these are defined in your config.py
    key_labels = [f"Key_{i}" for i in range(n_keys)]  # Replace with actual labels
    click_labels = [f"Click_{i}" for i in range(n_clicks)]  # Replace with actual labels

    print("\n--- Key Press Statistics ---")
    for i, count in enumerate(global_key_counts):
        percent = (count / global_total_frames) * 100 if global_total_frames > 0 else 0
        print(f"Key {key_labels[i]}: {percent:.2f}% pressed")

    print("\n--- Click Statistics ---")
    for i, count in enumerate(global_click_counts):
        percent = (count / global_total_frames) * 100 if global_total_frames > 0 else 0
        print(f"Click {click_labels[i]}: {percent:.2f}% pressed")

    print("\n--- Yaw Bin Distribution ---")
    for i, count in enumerate(global_yaw_counts):
        percent = (count / global_total_frames) * 100 if global_total_frames > 0 else 0
        print(f"Yaw bin {mouse_x_bins[i]}: {percent:.2f}%")

    print("\n--- Pitch Bin Distribution ---")
    for i, count in enumerate(global_pitch_counts):
        percent = (count / global_total_frames) * 100 if global_total_frames > 0 else 0
        print(f"Pitch bin {mouse_y_bins[i]}: {percent:.2f}%")