from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List, Optional
import time
import uvicorn

from config import *
from train import crit_mse, custom_loss, wasd_acc, space_acc, Lclk_acc, Rclk_acc, m_x_acc, m_y_acc


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
            print("TFLite model loaded")
        else:
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
            self.input_shape = self.model.input_shape[1:]  # Skip batch dimension
            print("Regular TensorFlow model loaded")

    def predict(self, input_data: np.ndarray):
        if self.lite:
            if input_data.shape != self.input_shape:
                input_data = np.resize(input_data, self.input_shape)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            # Add batch dimension for regular model
            input_data = np.expand_dims(input_data, axis=0)
            if input_data.shape[1:] != self.input_shape:
                input_data = np.resize(input_data, (1,) + self.input_shape)
            return self.model.predict(input_data, verbose=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model based on Lite parameter
    app.state.model_wrapper = ModelWrapper(lite=app.state.lite_flag)

    # Warming up the model
    print("Warming up the model with a dummy input...")
    try:
        dummy_input = np.zeros(app.state.model_wrapper.input_shape, dtype=np.float32)
        _ = app.state.model_wrapper.predict(dummy_input)
        print("Warm-up complete.")
    except Exception as e:
        print("Warm-up failed:", str(e))
    yield

    # Cleanup if needed
    if hasattr(app.state, 'model_wrapper'):
        del app.state.model_wrapper


app = FastAPI(lifespan=lifespan)


# Input model
class FrameInput(BaseModel):
    frame: List[List[List[List[List[float]]]]]
    lite: Optional[bool] = None  # Optional parameter to override default


@app.on_event("startup")
async def startup_event():
    # Set default mode (True for Lite, False for regular)
    # This will be used if not overridden by the request
    app.state.lite_flag = True  # Default to Lite mode


@app.post("/predict")
async def predict(data: FrameInput):
    try:
        start_time = time.time()

        # Use request-specific mode if provided, otherwise use default
        use_lite = data.lite if data.lite is not None else app.state.lite_flag

        # Convert input to numpy array and normalize
        frame_np = np.array(data.frame, dtype=np.float32) / 255.0

        # Get predictions (using the pre-loaded model)
        output_data = app.state.model_wrapper.predict(frame_np)

        elapsed_time = (time.time() - start_time) * 1000  # milliseconds
        print(f"Inference time: {elapsed_time:.2f} ms")

        pred_actions = output_data[0][0].tolist()
        return {
            "actions": pred_actions,
            "model_type": "TFLite" if app.state.model_wrapper.lite else "Regular TensorFlow"
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Change this to use either the lite model or the regular model
    app.state.lite_flag = False

    uvicorn.run(app, host="0.0.0.0", port=8000)