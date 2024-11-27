import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Параметры
batch_size = 8
image_size = (128, 128)  # Размер изменён на 128x128
train_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/train/'
val_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/val/'
initial_learning_rate = 0.001  # Понижен для стабильности

# Функция для загрузки изображений
def load_images(directory, target_size=(128, 128)):
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
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
        )
    except RuntimeError as e:
        print(e)

# Упрощённая модель UNET
def unet_autoencoder(input_size=(128, 128, 3)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(p2)

    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c3)
    c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(u4)

    u5 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c4)
    c5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(u5)

    outputs = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01))(c5)

    model = tf.keras.models.Model(inputs, outputs)

    return model

# Создание модели
model = unet_autoencoder(input_size=(128, 128, 3))

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['accuracy'])

# Тренировка модели
with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

# Построение графиков потерь
train_losses = history.history['loss']
val_losses = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Потери на обучении')
plt.plot(val_losses, label='Потери на валидации')
plt.title('Кривая обучения: Потери на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.savefig('metrics.png')  # Сохраняем график
plt.show()

