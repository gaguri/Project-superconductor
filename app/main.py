import io

import numpy as np
from PIL import Image
from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import keras
import uvicorn
from keras_preprocessing.image import load_img, img_to_array

app = FastAPI()
templates = Jinja2Templates(directory='templates')


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})


@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    model = keras.models.load_model('../UNET/unet_model.keras')
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image = image.resize((128, 128))
    image.save('./temp/temp_file.png')
    np_image = img_to_array(image) / 255.0
    predicted_params = model.predict(np.array([np_image])).tolist()[0]
    print(predicted_params)
    return {'params': {
        'D': predicted_params[0],
        'V': predicted_params[1],
        'td': predicted_params[2],
        'tp': predicted_params[3]
    }}


app.mount('/static', StaticFiles(directory='static'), 'static')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)
