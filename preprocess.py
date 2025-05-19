import numpy as np
import os
from tqdm import tqdm

from config import *

class ScreendataPreprocessor:
    def __init__(self, path, output_dir):
        self.path = path
        self.output_dir = output_dir
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.n_pixels = self.frame_width * self.frame_height
        self.n_keys = n_keys
        self.n_clicks = n_clicks
        self.mouse_x_bins = mouse_x_bins
        self.mouse_y_bins = mouse_y_bins
        self.n_mouse_x = len(self.mouse_x_bins)
        self.n_mouse_y = len(self.mouse_y_bins)
        self.total_label_dim = self.n_keys + self.n_clicks + self.n_mouse_x + self.n_mouse_y + 1
        self.marker_byte = marker_byte
        self.frame_size = self.n_pixels + self.n_keys + self.n_clicks + 2 + 1
        self.n_timesteps = n_timesteps

        os.makedirs(self.output_dir, exist_ok=True)

    def one_hot(self, index, length):
        vec = np.zeros(length, dtype=np.float32)
        if 0 <= index < length:
            vec[index] = 1.0
        return vec

    def process(self, name):
        with open(self.path, 'rb') as f:
            data = f.read()

        total_frames = len(data) // self.frame_size
        print(f"Loaded {total_frames} frames.")

        X_seq = []
        Y_seq = []

        key_counts = np.zeros(self.n_keys, dtype=np.int64)
        click_counts = np.zeros(self.n_clicks, dtype=np.int64)
        yaw_bin_counts = np.zeros(self.n_mouse_x, dtype=np.int64)
        pitch_bin_counts = np.zeros(self.n_mouse_y, dtype=np.int64)
        total_valid_frames = 0

        for i in tqdm(range(total_frames - self.n_timesteps)):
            frames = []
            labels = []

            for t in range(self.n_timesteps):
                offset = (i + t) * self.frame_size

                frame_data = data[offset:offset + self.n_pixels]
                key_data = data[offset + self.n_pixels : offset + self.n_pixels + self.n_keys]
                click_data = data[offset + self.n_pixels + self.n_keys : offset + self.n_pixels + self.n_keys + self.n_clicks]
                yaw_bin = data[offset + self.n_pixels + self.n_keys + self.n_clicks]
                pitch_bin = data[offset + self.n_pixels + self.n_keys + self.n_clicks + 1]
                marker = data[offset + self.frame_size - 1]

                if marker != self.marker_byte:
                    print(f"Frame {i+t} has invalid marker. Skipping sequence.")
                    print(f"Expected marker: {self.marker_byte}, Found: {marker} at frame {i + t}")
                    break

                gray = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.frame_height, self.frame_width))
                rgb = np.stack([gray] * 3, axis=-1)

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
                    self.one_hot(yaw_bin, self.n_mouse_x),
                    self.one_hot(pitch_bin, self.n_mouse_y),
                    np.array([0.0], dtype=np.float32)
                ])
                labels.append(label)

            if len(frames) == self.n_timesteps:
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