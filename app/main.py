import io
import os

import numpy as np
from PIL import Image
from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import keras
from keras_preprocessing.image import img_to_array

app = FastAPI()
templates = Jinja2Templates(directory='templates')


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})


@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    path = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(path, os.pardir))

    model = keras.models.load_model(os.path.join(parent_dir, 'UNET', 'best_model.h5'))
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image = image.resize((128, 128))
    np_image = img_to_array(image, data_format = "channels_first") / 255.0
    predicted_params = model.predict(np.array([np_image])).tolist()[0]
    print(predicted_params)
    print(file.filename)
    return {'params': {
        'D': predicted_params[0],
        'V': predicted_params[1],
        'td': predicted_params[2],
        'tp': predicted_params[3]
    }}


app.mount('/static', StaticFiles(directory='static'), 'static')