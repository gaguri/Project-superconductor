import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Параметры
batch_size = 8
image_size = (512, 512)
train_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/train/'
val_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/val/'
initial_learning_rate = 0.1

# Функция для загрузки изображений
def load_images(directory, target_size=(512, 512)):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Загрузка и нормализация данных
print("Загрузка тренировочных изображений...")
train_images = load_images(train_dir, target_size=image_size) / 255.0
print("Загрузка валидационных изображений...")
val_images = load_images(val_dir, target_size=image_size) / 255.0

# Проверка данных
print(f"Train images shape: {train_images.shape}")
print(f"Validation images shape: {val_images.shape}")

# Создание tf.data.Dataset
def create_dataset(images, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, images))  # x и y одинаковы для автоэнкодера
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_images, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(val_images, batch_size=batch_size, shuffle=False)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Ограничиваем память до 80% доступной
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
        )
    except RuntimeError as e:
        print(e)

# Определение модели UNET как автоэнкодера
def unet_autoencoder(input_size=(512, 512, 3)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c9)

    model = tf.keras.models.Model(inputs, outputs)

    return model

# Создание модели
model = unet_autoencoder(input_size=(512, 512, 3))

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['accuracy'])

# Тренировка модели
with tf.device('/GPU:0'):  # Явное указание использовать GPU
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
