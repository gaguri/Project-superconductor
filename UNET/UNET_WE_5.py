import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Параметры
batch_size = 8
image_size = (128, 128)
train_csv = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/csv_files/train_data.csv'
val_csv = r'C:/Users/bezzz/OneDrive/Desktop/All/Acheba/Hope/Project/csv_files/val_data.csv'
initial_learning_rate = 0.001

# Загрузка предварительно обученных весов
encoder_weights_file = 'encoder_weights.npy'
encoder_weights = np.load(encoder_weights_file, allow_pickle=True).item()

# Функция для загрузки изображений
def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    return img_to_array(img, data_format="channels_first") / 255.0

# Функция для загрузки данных из CSV
def load_data_from_csv(csv_path, target_size=(128, 128)):
    data = pd.read_csv(csv_path)
    images = []
    params = []
    image_paths = []
    for _, row in data.iterrows():
        img_path = row['Path']
        img = load_image(img_path, target_size=target_size)
        images.append(img)
        params.append([row['D'], row['V'], row['tb'], row['tp']])
        image_paths.append(img_path)
    return np.array(images), np.array(params), image_paths

# Загрузка данных
train_images, train_params, train_paths = load_data_from_csv(train_csv, target_size=image_size)
val_images, val_params, val_paths = load_data_from_csv(val_csv, target_size=image_size)

# Создание tf.data.Dataset
def create_dataset(images, params, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": images}, params))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_images, train_params, batch_size=batch_size)
val_dataset = create_dataset(val_images, val_params, batch_size=batch_size, shuffle=False)

def unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=None):
    image_input = tf.keras.layers.Input(shape=input_image_shape, name="image")
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_1")(image_input)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_2")(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c2)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_3")(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c3)
    bottleneck = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="bottleneck")(p3)
    flatten = tf.keras.layers.Flatten()(bottleneck)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    output = tf.keras.layers.Dense(4, activation='linear', name="output")(dense2)

    model = tf.keras.models.Model(inputs=image_input, outputs=output)

    if encoder_weights is not None:
        for layer in model.layers:
            if layer.name in encoder_weights:
                weight_shape_model = [w.shape for w in layer.get_weights()]
                weight_shape_loaded = [w.shape for w in encoder_weights[layer.name]]
                if weight_shape_model == weight_shape_loaded:
                    layer.set_weights(encoder_weights[layer.name])
                else:
                    print(f"Shape mismatch for layer {layer.name}: "f"model expects {weight_shape_model}, weights provided {weight_shape_loaded}")
    
    return model

# Загрузка модели
encoder_weights = np.load("encoder_weights.npy", allow_pickle=True).item()
model = unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=encoder_weights)


# Создание модели
model = unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=encoder_weights)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['mae'])

# Обучение модели
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint("best_model_with_l2.h5", save_best_only=True)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# График функции потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Оценка модели на тестовых данных
print("Оценка модели на тестовых данных...")
test_loss, test_mae = model.evaluate(val_dataset)  # Используем валидационные данные, либо загрузите тестовые
print(f"Validation Loss: {test_loss}, Validation MAE: {test_mae}")

# Предсказание на тестовых данных
print("Предсказание параметров...")
predictions = model.predict(val_dataset)

# Сохранение предсказанных параметров в CSV файл
print("Сохранение результатов предсказаний...")
predicted_df = pd.DataFrame(predictions, columns=["D_pred", "V_pred", "tb_pred", "tp_pred"])
predicted_df["Path"] = val_paths  # Добавляем пути изображений из данных валидации

# Сохранение в CSV
output_csv = "predicted_params_with_paths.csv"
predicted_df.to_csv(output_csv, index=False)

print(f"Предсказанные параметры сохранены в '{output_csv}'")
