import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import transforms

class PreprocessedDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Путь к CSV-файлу с путями к изображениям и параметрами.
        """
        self.data = pd.read_csv(csv_file)  # Загружаем CSV-файл
        self.transform = transforms.ToTensor()  # Определяем трансформацию

    def __len__(self):
        # Возвращаем количество строк в CSV
        return len(self.data)
    
    def __getitem__(self, idx):
        # Извлекаем путь к изображению
        img_path = self.data.iloc[idx, 0]  # Первая колонка — путь к изображению
        # Извлекаем параметры (все, кроме первой колонки)
        parameters = self.data.iloc[idx, 1:].values.astype('float32')

        # Загружаем изображение (оно уже предобработано)
        image = Image.open(img_path).convert('RGB')  # Конвертация в RGB на всякий случай
        image = self.transform(image)  # Преобразуем в тензор [C, H, W]
        # Возвращаем изображение и параметры
        return image, torch.tensor(parameters)
