import os.path

import keras
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from UNET.dataset_common_functions import get_test_dataset
import pandas as pd

batch_size = 16
dataset_directory = r'C:\Users\Egor\Desktop\study\project\Project-superconductor\unet_dataset'


def calculate_metrics_for_params(index, param_name):
    test = [i[index] for i in test_params]
    predictions = [i[index] for i in array]
    mse = mean_squared_error(test, predictions)
    rmse = root_mean_squared_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    print(f"MSE {mse:4f}, RMSE {rmse:4f}, MAPE {mape * 100:2f}% for parameter {param_name}")


def calculate_metrics_for_param(test_params, predicted_params, index):
    mapes = []
    for i in range(len(test_params)):
        mapes.append(round(mean_absolute_percentage_error(np.array([test_params[i][index]]),
                                                          np.array([predicted_params[i][index]])) * 100, 4))
    return mapes


def print_mape_parameters(df, column_name):
    max_mape = df[column_name].max()
    avg_mape = df[column_name].mean()
    min_mape = df[column_name].min()

    print(f'Минимальная относительная ошибка {column_name}, %: {min_mape:.4f}')
    print(f'Максимальная относительная ошибка {column_name}, %: {max_mape:.4f}')
    print(f'Средняя относительная ошибка {column_name}, %: {avg_mape:.4f}')
    print()


# Инициализация модели
print('Загрузка модели')
model = keras.models.load_model('unet_model.keras')

test_params, test_images, test_paths = get_test_dataset(dataset_directory)
predicted_params = []
for image in test_images:
    predicted_params.append(model.predict(np.array([image])).tolist()[0])

predicted_df = pd.DataFrame(predicted_params, columns=["D_pred", "V_pred", "tb_pred", "tp_pred"])
predicted_df["Path"] = test_paths  # Добавляем пути изображений

array = np.array(predicted_params)
mse = mean_squared_error(test_params, array)
rmse = root_mean_squared_error(test_params, array)
mape = mean_absolute_percentage_error(test_params, array)

print(f"Common MSE {mse:4f}")
print(f"Common RMSE {rmse:4f}")
print(f"Common MAPE {mape * 100:2f}%")

calculate_metrics_for_params(0, 'D')
calculate_metrics_for_params(1, 'V')
calculate_metrics_for_params(2, 'tb')
calculate_metrics_for_params(3, 't')

predicted_df["D_mape"] = calculate_metrics_for_param(test_params, predicted_params, 0)
predicted_df["V_mape"] = calculate_metrics_for_param(test_params, predicted_params, 1)
predicted_df["tb_mape"] = calculate_metrics_for_param(test_params, predicted_params, 2)
predicted_df["tp_mape"] = calculate_metrics_for_param(test_params, predicted_params, 3)
print_mape_parameters(predicted_df, 'D_mape')
print_mape_parameters(predicted_df, 'V_mape')
print_mape_parameters(predicted_df, 'tb_mape')
print_mape_parameters(predicted_df, 'tp_mape')

predicted_df.to_csv(os.path.join(dataset_directory, 'predicted_params_with_paths.csv'), index=False)
