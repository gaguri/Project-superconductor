import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

# Определяем наследующий класс от torch.utils.data.Dataset для подготовки датасета
class PreprocessedDataset(Dataset):
    def __init__(self, csv_file):
        # На вход подается путь к csv файлу с путями к изображениям и параметрам

        self.data = pd.read_csv(csv_file)       # Загружаем CSV-файл
        self.transform = transforms.ToTensor()  # Определяем трансформацию

    def __len__(self):
        
        # Возвращаем количество строк в CSV
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # Извлекаем путь к изображению из первой колонки CSV
        img_path = self.data.iloc[idx, 0]
        
        # Извлекаем параметры (все, кроме первой колонки), конвертируем в float32
        parameters = self.data.iloc[idx, 1:].values.astype('float32')

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')  # Конвертация в RGB на всякий случай
        image = self.transform(image)                # Преобразуем в тензор
        
        # Возвращаем изображение и параметры как тензор
        return image, torch.tensor(parameters)
