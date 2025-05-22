import numpy as np
import os
import matplotlib.pyplot as plt
from config import *


def interpret_actions(actions):
    """Convert raw action array to human-readable dictionary"""
    action_dict = {
        'W': actions[0] > 0.5,
        'A': actions[1] > 0.5,
        'S': actions[2] > 0.5,
        'D': actions[3] > 0.5,
        'Space': actions[4] > 0.5,
        'LeftClick': actions[5] > 0.5,
        'RightClick': actions[6] > 0.5
    }

    yaw_start = n_keys + n_clicks
    pitch_start = yaw_start + n_mouse_x

    yaw_bin = np.argmax(actions[yaw_start:pitch_start])
    action_dict['Yaw'] = mouse_x_bins[yaw_bin]

    pitch_bin = np.argmax(actions[pitch_start:pitch_start + n_mouse_y])
    action_dict['Pitch'] = mouse_y_bins[pitch_bin]

    return action_dict


def visualize_preprocessed_data(input_dir, output_dir, name, sequence_idx=0, num_frames=5, prediction=None):
    """
    Visualize preprocessed frames with their corresponding actions

    Args:
        input_dir: Directory containing input .npy files
        output_dir: Directory containing output .npy files
        name: Base filename (without _X.npy/_Y.npy)
        sequence_idx: Which sequence to visualize
        num_frames: How many frames to show
        prediction: The actions predicted by the model 
    """

    # Load data
    X = np.load(os.path.join(input_dir, f"{name}_X.npy"))
    Y = np.load(os.path.join(output_dir, f"{name}_Y.npy"))

    total_sequences = len(X)
    print(f"Visualizing Sequence {sequence_idx + 1}/{total_sequences} (X shape: {X.shape}, Y shape: {Y.shape})")

    # Get the specific sequence
    frames = X[sequence_idx]  # (seq_len, height, width, channels)
    actions = Y[sequence_idx]  # (seq_len, action_dim)

    for frame_idx in range(min(num_frames, len(frames))):
        # Create figure
        plt.figure(figsize=(16, 8))

        # Display the frame
        plt.subplot(1, 2, 1)
        frame = frames[frame_idx]
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)
        plt.imshow(frame)
        plt.title(f"Sequence {sequence_idx + 1}/{total_sequences} - Frame {frame_idx + 1}/{len(frames)}\n{frame.shape[1]}x{frame.shape[0]}")
        plt.axis('off')

        # Display actions
        plt.subplot(1, 2, 2)
        action_dict = interpret_actions(actions[frame_idx])

        check = '✓'
        cross = '✗'

        action_text = "Action Breakdown:\n\n"

        if prediction is None:
            # Show only actual (real) actions
            action_text += "Movement:\n"
            action_text += f"  W: {check if action_dict['W'] else cross}  "
            action_text += f"A: {check if action_dict['A'] else cross}  "
            action_text += f"S: {check if action_dict['S'] else cross}  "
            action_text += f"D: {check if action_dict['D'] else cross}\n\n"

            action_text += "Actions:\n"
            action_text += f"  SPACE: {check if action_dict['Space'] else cross}  "
            action_text += f"L-CLICK: {check if action_dict['LeftClick'] else cross}  "
            action_text += f"R-CLICK: {check if action_dict['RightClick'] else cross}\n\n"

            action_text += "Mouse Look:\n"
            action_text += f"  Yaw: {action_dict['Yaw']}°\n"
            action_text += f"  Pitch: {action_dict['Pitch']}°"

        else:
            # Show real vs predicted actions side-by-side
            pred_action_dict = interpret_actions(prediction)

            action_text += "Movement:\n"
            for key in ['W', 'A', 'S', 'D']:
                real = check if action_dict[key] else cross
                pred = check if pred_action_dict[key] else cross
                action_text += f"  {key}: {real} / {pred}  "
            action_text += "\n\n"

            action_text += "Actions:\n"
            for key, label in [('Space', 'SPACE'), ('LeftClick', 'L-CLICK'), ('RightClick', 'R-CLICK')]:
                real = check if action_dict[key] else cross
                pred = check if pred_action_dict[key] else cross
                action_text += f"  {label}: {real} / {pred}  "
            action_text += "\n\n"

            action_text += "Mouse Look:\n"
            action_text += f"  Yaw: {action_dict['Yaw']}° / {pred_action_dict['Yaw']}°\n"
            action_text += f"  Pitch: {action_dict['Pitch']}° / {pred_action_dict['Pitch']}°"

        plt.text(0.1, 0.5, action_text, fontsize=14, family='monospace')
        plt.axis('off')

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    name = "screendata_2025-05-11_10-12-14-155"
    input_dir = os.path.join(preprocessed_dataset_dir, "input")
    output_dir = os.path.join(preprocessed_dataset_dir, "output")

    NUM_SEQUENCES = 20

    for sequence_idx in range(0, NUM_SEQUENCES):
        visualize_preprocessed_data(
            input_dir=input_dir,
            output_dir=output_dir,
            name=name,
            sequence_idx=sequence_idx,
            num_frames=1
        )