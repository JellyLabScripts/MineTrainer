from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List
import time
import uvicorn

from config import *

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=lite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warming up the model
    print("Warming up the model with a dummy input...")
    try:
        dummy_shape = tuple(input_details[0]['shape'])
        dummy_input = np.zeros(dummy_shape, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        print("Warm-up complete.")
    except Exception as e:
        print("Warm-up failed:", str(e))

    yield

app = FastAPI(lifespan=lifespan)

# Input model
class FrameInput(BaseModel):
    frame: List[List[List[List[List[float]]]]]  # Shape: (1, 96, 150, 280, 3)

@app.post("/predict")
async def predict(data: FrameInput):
    try:
        start_time = time.time()

        frame_np = np.array(data.frame, dtype=np.float32) / 255.0

        if frame_np.shape != tuple(input_details[0]['shape']):
            print(f"Reshaping input from {frame_np.shape} to {input_details[0]['shape']}")
            frame_np = np.resize(frame_np, input_details[0]['shape'])

        interpreter.set_tensor(input_details[0]['index'], frame_np)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        elapsed_time = (time.time() - start_time) * 1000  # milliseconds
        print(f"Inference time: {elapsed_time:.2f} ms")

        pred_actions = output_data[0][0].tolist()
        return {"actions": pred_actions}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)