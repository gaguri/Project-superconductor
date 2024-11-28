import os
import csv
import pandas as pd

# Указываем пути
folder_path = '/home/whytech/project/result'       # Папка с файлами
file_name = '/home/whytech/project/name_files.csv'  # Файл с названиями
file_path = '/home/whytech/project/path_files.csv'  # Файл с путями

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
name = pd.read_csv('/home/whytech/project/name_files.csv', header=None, sep = '_')

# Удаляем ненужные колонки
name = name.drop(name.columns[[0, 5, 6, 7, 8]], axis=1)

# Присваиваем новые имена колонкам
name.columns = ['D', 'V', 'tb', 'tp']

# Удаляем буквы из всех строк в DataFrame
name = name.apply(lambda x: x.str.replace(r'[A-Za-z]', '', regex=True))

# Читаем csv-файл с путями
path = pd.read_csv('/home/whytech/project/path_files.csv', header=None)

# Приваиваем колонке новое имя
path.columns = ['Path']

# Объединяем два датасета
dataset = pd.concat([path, name], axis=1)

# Сохраняем полученный датафрейм
csv_file_path = '/home/whytech/project/labels.csv'
dataset.to_csv(csv_file_path, index=False)
 
# Загружаем CSV-файл 
data = pd.read_csv('/home/whytech/project/labels.csv') 
 
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
train_data.to_csv('/home/whytech/project/train_data.csv', index=False, header=True) 
val_data.to_csv('/home/whytech/project/val_data.csv', index=False, header=True)
test_data.to_csv('/home/whytech/project/test_data.csv', index=False, header=True)
