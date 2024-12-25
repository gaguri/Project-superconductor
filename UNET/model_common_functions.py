import tensorflow as tf
from tensorflow.keras import regularizers


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