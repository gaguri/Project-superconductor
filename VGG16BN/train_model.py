import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import PreprocessedDataset
from VGG16BN import modified_model
from torch.utils.data import DataLoader

# Инициализация модели
model = modified_model()
criterion = nn.MSELoss()                                # Функция потерь для регрессии
optimizer = optim.Adam(model.parameters(), lr=1e-4)     # Оптимизатор

# Переход на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Данные для обучения и валидации
dataset_train = PreprocessedDataset(csv_file='/home/whytech/project/train_data.csv')
dataloader_train = DataLoader(dataset_train, batch_size=16)
dataset_val = PreprocessedDataset(csv_file='/home/whytech/project/val_data.csv')
dataloader_val = DataLoader(dataset_val, batch_size=16)

# Цикл обучения
num_epochs = 200  
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    
    # Обучение модели
    model.train()  # Перевод в режим обучения
    train_loss = 0.0
    
    for batch_images, batch_params in dataloader_train:
        
        batch_images = batch_images.to(device)
        batch_params = batch_params.to(device)

        optimizer.zero_grad() # Обнуление градиентов
        
        # Прямой проход
        outputs = model(batch_images)  # Предсказания модели       
        loss = criterion(outputs, batch_params) # Вычисление функции потерь
        
        # Обратный проход 
        loss.backward()
        optimizer.step() # Оптимизация
        
        # Сохраняем значение потерь
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(dataloader_train))

    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
         for batch_images, batch_params in dataloader_val:
            batch_images = batch_images.to(device)
            batch_params = batch_params.to(device)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_params)
            val_loss += loss.item()

    val_losses.append(val_loss / len(dataloader_val))

    # Печать результатов за эпоху
    print(f"Эпоха [{epoch + 1}/{num_epochs}],\
        Потеря при обучении: {train_losses[-1]:.4f},\
        Потеря при валидации: {val_losses[-1]:.4f}")

weights_path = '/home/whytech/project/model_weights.pth'
torch.save(model.state_dict(), weights_path)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Потери на обучении')
plt.plot(val_losses, label='Потери на валидации')
plt.title('Кривая обучения: Потери на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.savefig('/home/whytech/project/metrics.png')
