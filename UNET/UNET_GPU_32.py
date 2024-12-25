import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from UNET.dataset_common_functions import get_train_dataset, get_validation_dataset
from UNET.model_common_functions import unet_full_model

# Параметры
batch_size = 8
image_size = (128, 128)  # Размер изображений
initial_learning_rate = 0.001  # Понижен для стабильности
dataset_path = r'C:\Users\Egor\Desktop\study\project\Project-superconductor\unet_dataset'

# Загрузка данных
print("Загрузка тренировочных данных...")
train_params, train_images, train_paths = get_train_dataset(dataset_path)
print("Загрузка валидационных данных...")
val_params, val_images, val_paths = get_validation_dataset(dataset_path)

# Проверка данных
print(f"Train images shape: {train_images.shape}, Train params shape: {train_params.shape}")
print(f"Validation images shape: {val_images.shape}, Validation params shape: {val_params.shape}")


# Создание tf.data.Dataset
def create_dataset(images, params, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": images}, params))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = create_dataset(train_images, train_params, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(val_images, val_params, batch_size=batch_size, shuffle=False)


# Оптимизированная модель

model = unet_full_model(input_image_shape=(3, 128, 128))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['mae'])

# Обучение модели
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", save_best_only=True, save_weights_only=True)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks,
    verbose=1
)

# График функции потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

print("Обучение завершено, лучшая модель сохранена в 'best_model.h5'")
