import io
import tensorflow as tf
from tensorflow.keras import regularizers

import numpy as np
from PIL import Image
from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from keras_preprocessing.image import img_to_array

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# костыль, вынестив в общую функциюб создание модели
def unet_full_model(input_image_shape=(3, 128, 128)):
    image_input = tf.keras.layers.Input(shape=input_image_shape, name="image")
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first")(image_input)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first")(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c2)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c3)
    bottleneck = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(p3)
    flatten = tf.keras.layers.Flatten()(bottleneck)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    output = tf.keras.layers.Dense(4, activation='linear', name="output")(dense2)
    return tf.keras.models.Model(inputs=image_input, outputs=output)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})


@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    model = unet_full_model()
    model.load_weights('../UNET/best_model.h5')
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