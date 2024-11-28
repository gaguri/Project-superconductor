from PIL import Image
import os

input_dir = '/home/whytech/project/result'
output_dir = '/home/whytech/project/result_resized'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    img_resized = img.resize((224, 224))  # Изменяем размер
    img_resized.save(os.path.join(output_dir, filename))  # Сохраняем
