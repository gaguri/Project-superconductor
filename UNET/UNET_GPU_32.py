import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

from UNET.dataset_common_functions import get_train_dataset, get_validation_dataset

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


# Упрощённая модель UNET с сжатием до 4x4 и замороженным декодером
def unet_with_dense(input_image_shape=(128, 128, 3)):
    # Вход для изображения
    image_input = tf.keras.layers.Input(shape=input_image_shape, name="image")

    # Encoder с L2 регуляризацией
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01))(image_input)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01))(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01))(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01))(p3)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)  # Это сжатие до 8x8

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01))(p4)
    p5 = tf.keras.layers.MaxPooling2D((2, 2))(c5)  # Это сжатие до 4x4

    # Bottleneck с L2 регуляризацией
    bottleneck = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                                        kernel_regularizer=regularizers.l2(0.01))(p5)

    # Декодер
    d1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(bottleneck)
    d2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(d1)
    d3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(d2)
    d4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(d3)
    d5 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(d4)

    # Замораживаем декодер
    for layer in [d1, d2, d3, d4, d5]:
        layer.trainable = False

    # Global Average Pooling
    gap = tf.keras.layers.GlobalAveragePooling2D()(bottleneck)

    # Fully Connected слой для предсказания параметров с L2 регуляризацией и Dropout
    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(gap)
    dense1_drop = tf.keras.layers.Dropout(0.5)(dense1)  # Dropout для предотвращения переобучения
    dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense1_drop)
    dense2_drop = tf.keras.layers.Dropout(0.5)(dense2)  # Dropout для предотвращения переобучения
    output = tf.keras.layers.Dense(4, activation='linear', name="output")(dense2_drop)

    model = tf.keras.models.Model(inputs=image_input, outputs=output)
    return model


# Создание модели
model = unet_with_dense(input_image_shape=(128, 128, 3))

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['mae'])

# Печать структуры модели
model.summary()

# Тренировка модели
with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        epochs=200,
        validation_data=val_dataset,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    )

# Построение графиков потерь
train_losses = history.history['loss']
val_losses = history.history['val_loss']
model.save('unet_model.keras')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Потери на обучении')
plt.plot(val_losses, label='Потери на валидации')
plt.title('Кривая обучения: Потери на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.savefig('metrics_with_params.png')  # Сохраняем график
plt.show()
