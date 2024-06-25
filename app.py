from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from starlette.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = load_model('my_model (1).h5')  # Load your trained model

class_names = {0: "Alluvial Soil", 1: "Black Soil", 2: "Clay Soil", 3: "Red Soil"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)

    img = image.load_img(file.filename, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    # Threshold confidence level for prediction
    threshold = 0.5  # Adjust this value based on your requirements

    if confidence >= threshold and predicted_class in class_names:
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = "Unknown"

    return {"class": predicted_label, "confidence": float(confidence)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return """
    <html>
        <body>
            <h1>Upload an image</h1>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
