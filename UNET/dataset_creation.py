from UNET.dataset_common_functions import create_and_save_dataset

#Пересоздание датасета из папки, в которой лежат исходные csv файлы и в которой есть директория result с сгенерированными изображениями в папку output_dataset_directory отукда при обучении и тестировании буду браться файлы
csvs_directory = r'C:\Users\Egor\Desktop\study\project\cube4-uniform-part1'
output_dataset_directory = r'C:\Users\Egor\Desktop\study\project\Project-superconductor\unet_dataset'
create_and_save_dataset(csvs_directory, output_dataset_directory)