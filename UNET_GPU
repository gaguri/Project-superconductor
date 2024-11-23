import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from keras.callbacks import Callback
import matplotlib.pyplot as plt

# Проверка наличия GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU found, using CPU instead.")
else:
    print(f"Using GPU: {physical_devices[0]}")

'''# Установка памяти GPU для использования по требованию (не полный захват)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)'''

# Параметры
batch_size = 8
image_size = (512, 512)  # Изменим размер изображений на 512x512
train_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/train/'
val_dir = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/Data/val/'
train_losses = []
val_losses = []

# Функция для загрузки изображений
def load_images_from_directory(directory, target_size=(512, 512)):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Загружаем изображения для тренировочного и валидационного датасетов
train_images = load_images_from_directory(train_dir, target_size=image_size)
val_images = load_images_from_directory(val_dir, target_size=image_size)

# Нормализация изображений
train_images = train_images / 255.0  # Нормализация для автоэнкодера
val_images = val_images / 255.0

# Создаём генератор с использованием flow
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_images, batch_size=batch_size)
val_generator = val_datagen.flow(val_images, val_images, batch_size=batch_size)

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

# Обновление потерь после каждой эпохи
class LossHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Добавляем текущие значения потерь в списки
        train_losses.append(logs.get('loss'))
        val_losses.append(logs.get('val_loss'))

# Создаём модель
model = unet_autoencoder(input_size=(512, 512, 3))

# Заморозка слоёв до определенного этапа (например, до encoder части)
for layer in model.layers[:10]:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

history_callback = LossHistory()
with tf.device('/GPU:0'):  # Явное указание на использование GPU
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[history_callback]
    )

plt.figure(figsize=(10, 5)) 
plt.plot(train_losses, label='Потери на обучении') 
plt.plot(val_losses, label='Потери на валидации') 
plt.title('Кривая обучения: Потери на обучении и валидации') 
plt.xlabel('Эпохи') 
plt.ylabel('Потери') 
plt.legend() 
plt.savefig('metrics.png')
plt.show()
