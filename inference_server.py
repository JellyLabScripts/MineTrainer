from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List
import uvicorn

from config import batch_size

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("saved_model/minecraft_behavior_model.keras", compile=False)
print("Model loaded")

# Data model
class FrameInput(BaseModel):
    frame: List[List[List[List[List[float]]]]]  # (1, 96, 150, 280, 3)

@app.post("/predict")
async def predict(data: FrameInput):
    try:
        frame_np = np.array(data.frame, dtype=np.float32) / 255.0  # Normalize
        preds = model.predict(frame_np)
        pred_actions = preds[0][0].tolist()  # Assuming output shape is (1, 1, 46)

        return {"actions": pred_actions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)