from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO #normal bir dosya gibi bytelarla işlem yapıyo
from PIL import Image #image okumak için
import tensorflow as tf

app= FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1")
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

    predictions= MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class" : predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)