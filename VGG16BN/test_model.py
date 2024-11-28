import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from dataset import PreprocessedDataset
from torch.utils.data import DataLoader
from VGG16BN import modified_model

model = modified_model()

model.load_state_dict(torch.load('/home/whytech/project/model_weights.pth', weights_only=True))

model.eval()
dataset_test = PreprocessedDataset(csv_file='/home/whytech/project/test_data.csv')
dataloader_test = DataLoader(dataset_test, batch_size=16)

all_predictions = []
all_labels = []
total_loss = 0.0
criterion = torch.nn.MSELoss()

# Тестирование модели
with torch.no_grad():  # Не вычисляем градиенты, потому что мы только тестируем
    for batch_images, batch_params in dataloader_test:

        # Передаем изображения через модель
        outputs = model(batch_images)

        # Рассчитываем потери (например, MSE)
        loss = criterion(outputs, batch_params)  # Предсказания модели и реальные параметры
        total_loss += loss.item()

        # Сохраняем предсказания и настоящие значения для последующего анализа
        all_predictions.append(outputs)
        all_labels.append(batch_params)

# Вычисляем среднее значение потерь
average_loss = total_loss / len(dataset_test)
print(f"Средняя потеря на тестовом наборе: {average_loss}")

# Преобразуем списки в тензоры для дальнейшего анализа
all_predictions = torch.cat(all_predictions, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Для регрессии, например, используем MSE
mse = mean_squared_error(all_labels.numpy(), all_predictions.numpy())
print(f"Среднеквадратичная ошибка (MSE): {mse}")

df = pd.DataFrame(all_predictions, columns=['Dpr', 'Vpr', 'tbpr', 'tppr'])
df.to_csv('/home/whytech/project/predictions.csv', index=False)
