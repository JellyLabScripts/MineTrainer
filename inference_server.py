from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List
import time
import uvicorn
import matplotlib.pyplot as plt


from config import *
from train import custom_loss, wasd_acc, space_acc, Lclk_acc, Rclk_acc, m_x_acc, m_y_acc


class ModelWrapper:
    def __init__(self):
        self.model = None
        print("Loading regular TensorFlow model...")
        self.model = tf.keras.models.load_model(
            regular_model_path,
            custom_objects={
                'custom_loss': custom_loss,
                'wasd_acc': wasd_acc,
                'space_acc': space_acc,
                'Lclk_acc': Lclk_acc,
                'Rclk_acc': Rclk_acc,
                'm_x_acc': m_x_acc,
                'm_y_acc': m_y_acc,
                'crit_mse': crit_mse
            }
        )
        self.input_shape = self.model.input_shape[1:]
        print("Regular TensorFlow model loaded")


    # Note that this includes the batch dimension but test_model doesn't
    def predict(self, input_data: np.ndarray):
        if input_data.shape != self.input_shape:
            print(f"Wrong input data shape! Found {input_data.shape} but the model requires {self.input_shape}")

        input_data = np.expand_dims(input_data, axis=0)
        return self.model.predict(input_data, verbose=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_wrapper = ModelWrapper()

    # Warming up the model
    print("Warming up the model with a dummy input...")
    try:
        dummy_input = np.zeros(app.state.model_wrapper.input_shape, dtype=np.float32)
        _ = app.state.model_wrapper.predict(dummy_input)
        print("Warm-up complete.")
    except Exception as e:
        print("Warm-up failed:", str(e))
    yield

    if hasattr(app.state, 'model_wrapper'):
        del app.state.model_wrapper


app = FastAPI(lifespan=lifespan)


# Input model
class FrameInput(BaseModel):
    frame: List[List[List[List[float]]]]


@app.post("/predict")
async def predict(data: FrameInput):
    try:
        start_time = time.time()

        frame_np = np.array(data.frame, dtype=np.float32) / 255.0
        # visualize_frame(frame_np[0])
        output_data = app.state.model_wrapper.predict(frame_np)

        elapsed_time = (time.time() - start_time) * 1000  # milliseconds
        print(f"Inference time: {elapsed_time:.2f} ms")

        pred_actions = output_data[0][0].tolist()
        return {
            "actions": pred_actions,
            "model_type": "Regular TensorFlow"
        }

    except Exception as e:
        return {"error": str(e)}



def visualize_frame(frame_data: np.ndarray):
    """
    Visualize a single frame received by the API with optional action information

    Args:
        frame_data: The frame data as a numpy array (height, width, channels)
    """
    plt.figure(figsize=(16, 8))

    # Display the frame
    plt.subplot(1, 2, 1)
    if frame_data.dtype == np.float32:
        frame = (frame_data * 255).astype(np.uint8)
    else:
        frame = frame_data

    plt.imshow(frame)
    plt.title(f"Received Frame\n{frame.shape[1]}x{frame.shape[0]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
