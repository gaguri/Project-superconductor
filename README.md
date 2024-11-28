# Project-superconductor

Файл image_generation.py генерирует изображения из исходных csv-файлов.  
Файл DataProcessor.py проходит по всем csv-файлам и с помощью image_generation.py генерирует изображения.  
Файл script_labels.py создает csv-файл с разметкой для обучения.  

## VGG16 BN  
В папке VGG16BN находятся следующие файлы:  
1. processing_data.py - подготовка изображений (сжатие с ~500х500 пикселей до 224х224), создание csv-файла с разметкой, разделение этого файла на 3 (train, val, test)
2. dataset.py - предподготовка датасета
3. VGG16BN.py - сама модель
4. train_model.py - обучение модели; после завершения обучения получаем график лоссов при обучении и валидации
5. test_model.py - теститрование модели; после завершения тестирования сохраняет csv-файл с предсказаниями
6. stats.py - расчет относительной ошибки определения; статистическая обработка результатов (min, max, mean) с выводом в консоль; создает csv-файл со следующими колонками:
   * истинные значения параметров (D, V, tb, tp)
   * предсказанные значения параметров (Dpr, Vpr, tbpr, tppr)
   * рассчитанные значения MAPE (MAPE_D, MAPE_V, MAPE_tb, MAPE_tp)

Запускать файлы в порядке: 1, 4, 5, 6

