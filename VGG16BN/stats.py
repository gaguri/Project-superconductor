import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Пути к файлам
true_result_path = '/home/whytech/project/csv_files/test_data.csv'      # Истинные результаты
pred_result_path = '/home/whytech/project/csv_files/predictions.csv'    # Предсказанные результаты
stat_path = '/home/whytech/project/csv_files/stat.csv'                  # Статистическая обработка параметров

def calculate_mape(true_values, pred_values):
    '''
    Функция для расчета MAPE

    Возвращает значение MAPE, округленное до четырех знаков
    '''
    return round(mean_absolute_percentage_error(true_values, pred_values) * 100, 4)

def add_mape_column(df, true_column, pred_column):

    # Создаем название для нового столбца MAPE
    mape_column_name = f'MAPE_{true_column}, %'

    # Рассчитываем MAPE и добавляем результаты в новый столбец
    df[mape_column_name] = df.apply(
        lambda row: calculate_mape([row[true_column]], [row[pred_column]]),
        axis=1
    )

# Чтение файлов
true_result = pd.read_csv(true_result_path)
pred_result = pd.read_csv(pred_result_path)

# Создаем новый датафрейм из реальных значений и предсказанных моделью
stat = pd.DataFrame({
    'D': true_result['D'],
    'Dpr': pred_result['Dpr'],

    'V': true_result['V'],
    'Vpr': pred_result['Vpr'],

    'tb': true_result['tb'],
    'tbpr': pred_result['tbpr'],

    'tp': true_result['tp'],
    'tppr': pred_result['tppr']
})

# Список параметров с именами колонок
parameters = [
    ('D', 'Dpr'),
    ('V', 'Vpr'),
    ('tb', 'tbpr'),
    ('tp', 'tppr')
]

# Считаем MAPE для каждого параметра
for true_col, pred_col in parameters:
    add_mape_column(stat, true_col, pred_col)

# Анализ результатов для каждого параметра
for true_col, pred_col in parameters:

    mape_column_name = f'MAPE_{true_col}, %'

    min_mape = stat[mape_column_name].min()
    max_mape = stat[mape_column_name].max()
    avg_mape = stat[mape_column_name].mean()
    
    print(f'Минимальная относительная ошибка {true_col}, %: {min_mape:.4f}')
    print(f'Максимальная относительная ошибка {true_col}, %: {max_mape:.4f}')
    print(f'Средняя относительная ошибка {true_col}, %: {avg_mape:.4f}')
    print()

# Сохранение обработки в файл
stat.to_csv(stat_path, index=False)