import json
import os.path
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def get_parameters_from_filename(filename):
    parts = filename.split('_')
    D = float(parts[1][1:])
    V = float(parts[2][1:])
    tb = float(parts[3][2:])
    tp = float(parts[4][2:])
    return D, V, tb, tp


def get_params_and_images(csv_directory_name, resulted_path, image_size=None):
    images_directory = os.path.join(csv_directory_name, "result")
    csvs_filenames = [file for file in os.listdir(csv_directory_name) if
                      os.path.isfile(os.path.join(csv_directory_name, file))]
    os.makedirs(resulted_path, exist_ok=True)
    data_params = []
    for filename in csvs_filenames:
        full_path = os.path.join(csv_directory_name, filename)
        without_extension = filename[:-4]
        D, V, tb, tp = get_parameters_from_filename(without_extension)
        png_filename = f'{without_extension}.png'
        image_full_path = os.path.join(images_directory, png_filename)
        img = Image.open(image_full_path)
        if image_size is not None:
            img = img.resize(image_size)
        resulted_image_full_path = os.path.join(resulted_path, png_filename)
        img.save(resulted_image_full_path)
        data_params.append({
            'csv_file_path': full_path,
            'img_full_path': resulted_image_full_path,
            'D_param': D,
            'V_param': V,
            'tb_param': tb,
            'tp_param': tp
        })
    return data_params


def create_and_save_dataset(csv_directory_name, output_directory):
    data = get_params_and_images(csv_directory_name, output_directory, (128, 128))
    random.shuffle(data)
    dataset_size = len(data)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    with open(os.path.join(output_directory, 'train_dataset_params.json'), 'w+', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False)

    with open(os.path.join(output_directory, 'validation_dataset_params.json'), 'w+', encoding='utf-8') as file:
        json.dump(val_data, file, ensure_ascii=False)

    with open(os.path.join(output_directory, 'test_dataset_params.json'), 'w+', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False)


def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    return img_to_array(img, data_format = "channels_first") / 255.0


def get_train_dataset(path):
    return get_dataset(os.path.join(path, 'train_dataset_params.json'))


def get_validation_dataset(path):
    return get_dataset(os.path.join(path, 'validation_dataset_params.json'))


def get_test_dataset(path):
    return get_dataset(os.path.join(path, 'test_dataset_params.json'))


def get_dataset(dataset_path):
    with open(dataset_path, 'r+', encoding='utf-8') as file:
        data = json.load(file)
    params = []
    images = []
    image_paths = []
    for item in data:
        images.append(load_image(item['img_full_path']))
        params.append([item['D_param'], item['V_param'], item['tb_param'], item['tp_param']])
        image_paths.append(item['img_full_path'])
    return (np.array(params), np.array(images), image_paths)
