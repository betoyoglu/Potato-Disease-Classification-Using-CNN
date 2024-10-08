from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO #normal bir dosya gibi bytelarla işlem yapıyo
from PIL import Image #image okumak için
import tensorflow as tf
import requests

app= FastAPI()

endpoint= "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Potato___Early_blight","Potato___Late_blight", "Potato___healthy"]

@app.get("/ping")
async def ping():
    return "hello i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) #byteları image olarak okuyacak sonra onu nparraye çevirecek
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...) #uploadfile is my data type yani foto yükliyceğimi söylüyorum 
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)

    json_data={
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class= CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class" : predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)