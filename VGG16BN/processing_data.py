from PIL import Image
import os
import csv
import pandas as pd

# Указываем пути
input_dir = '/home/whytech/project/result'                      # Путь к исходным изображениям
output_dir = '/home/whytech/project/result_resized'             # Путь к измененным изображениям

csv_files = '/home/whytech/project/csv_files'                   # Папка для хранения всех csv файлов
file_name = '/home/whytech/project/csv_files/name_files.csv'    # Файл с названиями изображений
file_path = '/home/whytech/project/csv_files/path_files.csv'    # Файл с путями до изображений

train_data_path = '/home/whytech/project/csv_files/train_data.csv'   # Файл с тренировочным датасетом
val_data_path = '/home/whytech/project/csv_files/val_data.csv'       # Файл с валидационным датасетом
test_data_path = '/home/whytech/project/csv_files/test_data.csv'     # Файл с тестовым датасетом

def resize_pics(input_dir, output_dir):
    '''
    Функция меняющая размеры изображения до 224х224 пикселей

    Аргументы:
            input_dir: путь к папке с исходными изображениями
            output_dir: путь к папке с измененными изображениями
    '''

    # Создаём папку для сохранения обработанных изображений
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по всем файлам в указанной папке
    for filename in os.listdir(input_dir):

        # Получаем полный путь к изображению
        img_path = os.path.join(input_dir, filename)

        # Открываем изображение
        img = Image.open(img_path)

        # Изменяем размер изображения до 224x224
        img_resized = img.resize((224, 224))

        # Сохраняем изменённое изображение в папке назначения
        img_resized.save(os.path.join(output_dir, filename))

def csv_name(output_dir):
    '''
    Функция для получения csv файла с названиями изображений

    Аргументы:
            output_dir: путь к папке с измененными изображениями
    '''
    
    # Получаем список файлов в папке
    files = os.listdir(output_dir)

    # Записываем названия файлов в CSV
    with open(file_name, mode='w', newline='') as file:
         writer = csv.writer(file)
         for filename in files:
            writer.writerow([filename])

def csv_path(output_dir):
    '''
    Функция для получения csv файла с путями изображений

    Аргументы:
            output_dir: путь к папке с измененными изображениями
    '''

    # Получаем список всех файлов в папке с их полными путями
    file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

    # Записываем пути в CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for path in file_paths:
            writer.writerow([path])

def create_labels(file_name, file_path):
    '''
    Функция для создания csv файла с разметкой

    Аргументы:
            file_name: csv файл с названиями изображений
            file_path: csv файл с путями изображений
    '''

    # Читаем csv-файл c названиями
    name = pd.read_csv(file_name, header=None, sep = '_')

    # Удаляем ненужные колонки
    name = name.drop(name.columns[[0, 5, 6, 7, 8]], axis=1)

    # Присваиваем новые имена колонкам
    name.columns = ['D', 'V', 'tb', 'tp']

    # Удаляем буквы из всех строк в DataFrame
    name = name.apply(lambda x: x.str.replace(r'[A-Za-z]', '', regex=True))

    # Читаем csv-файл с путями
    path = pd.read_csv(file_path, header=None)

    # Приваиваем колонке новое имя
    path.columns = ['Path']

    # Объединяем два датасета
    dataset = pd.concat([path, name], axis=1)

    # Сохраняем полученный датафрейм
    labels_path = '/home/whytech/project/csv_files/labels.csv'
    dataset.to_csv(labels_path, index=False)

    # Возвращаем путь к файлу с разметкой
    return labels_path

def split_data(labels_path):
    '''
    Функция для разделения датасета на train, test и val

    Аргументы:
            labels_path: csv файл с разметкой
    '''

    # Загружаем CSV-файл 
    data = pd.read_csv(labels_path) 
 
    # Извлекаем строки без заголовка (первая строка — заголовки) 
    data_rows = data.iloc[1:]  # Берём строки с 1 по 419 
 
    # Перемешиваем строки для случайного распределения 
    data_rows = data_rows.sample(frac=1, random_state=42).reset_index(drop=True) 
 
    train_size = int(0.6 * len(data_rows))
    val_size = int(0.2 * len(data_rows)) 

    train_data = data_rows[:train_size]
    val_data = data_rows[train_size:train_size + val_size]   
    test_data = data_rows[train_size + val_size:]    
 
    # Сохраняем данные в два CSV-файла 
    train_data.to_csv(train_data_path, index=False, header=True) 
    val_data.to_csv(val_data_path, index=False, header=True)
    test_data.to_csv(test_data_path, index=False, header=True)

# Создаём папку для сохранения csv файлов
os.makedirs(csv_files, exist_ok=True)

print(f'Начало работы с данными\n'
      f'Изменение размеров изображений')
print()

resize_pics(input_dir, output_dir)
print(f'Измененные изображения находятся в директории: {output_dir}')
print()

csv_name(output_dir)
print('Создан csv файл с названиями изображений')
print()

csv_path(output_dir)
print('Создан csv файл с путями до изображений')
print()

labels_path = create_labels(file_name, file_path)
print(f'Размеченные данные находятся в директории: {labels_path}')
print()

split_data(labels_path)
print(f'Датасет разделен на 3 файла: train_data, val_data, test_data\n'
      f'Файлы находятся в директории: {csv_files}')