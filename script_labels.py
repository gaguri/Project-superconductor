import os
import csv
import pandas as pd

# Указываем пути
folder_path = '/root/project/Dataset'       # Папка с файлами
file_name = '/root/project/name_files.csv'  # Файл с названиями
file_path = '/root/project/path_files.csv'  # Файл с путями

def csv_name(folder_path):

    # Получаем список файлов в папке
    files = os.listdir(folder_path)

    # Записываем названия файлов в CSV
    with open(file_name, mode='w', newline='') as file:
         writer = csv.writer(file)
         for filename in files:
            writer.writerow([filename])

def csv_path(folder_path):

    # Получаем список всех файлов в папке с их полными путями
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Записываем пути в CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for path in file_paths:
            writer.writerow([path])

csv_name(folder_path)
csv_path(folder_path)

# Читаем csv-файл c названиями
name = pd.read_csv('/root/project/name_files.csv', header=None, sep = '_')

# Удаляем ненужные колонки
name = name.drop(name.columns[[0, 5, 6, 7, 8]], axis=1)

# Присваиваем новые имена колонкам
name.columns = ['D', 'V', 'tb', 'tp']

# Удаляем буквы из всех строк в DataFrame
name = name.apply(lambda x: x.str.replace(r'[A-Za-z]', '', regex=True))

# Читаем csv-файл с путями
path = pd.read_csv('/root/project/path_files.csv', header=None)

# Приваиваем колонке новое имя
path.columns = ['Path']

# Объединяем два датасета
dataset = pd.concat([path, name], axis=1)

# Сохраняем полученный датафрейм
csv_file_path = '/root/project/labels.csv'
dataset.to_csv(csv_file_path, index=False)