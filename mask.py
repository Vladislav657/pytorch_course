import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import json

# --- 1. Загрузка модели SAM ---
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")  # Модель 'vit_b' или 'vit_h'
predictor = SamPredictor(sam)

# --- 2. Загрузка изображения ---
image_path = "1_59.jpg"  # Укажите путь к изображению
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image)
predictor.set_image(image)

# --- 3. Создание пустой маски (фон = 0) ---
mask = np.zeros(image.shape[:2], dtype=np.uint8)

with open("format.json") as f:
    data = json.load(f)
objects = data['train/1_59.jpg']

# --- 4. Обработка каждого объекта ---
for obj in sorted(objects, key=lambda x: x['category']):
    print(obj)
    bbox = obj['bbox']
    class_id = obj['category']

    # Преобразуем bbox в формат [x1, y1, x2, y2]
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    sam_bbox = np.array([x1, y1, x2, y2])

    # Генерация масок SAM для bbox
    masks, _, _ = predictor.predict(box=sam_bbox)

    # Выбор маски с площадью, наиболее близкой к заданной
    best_mask_idx = np.argmin([abs(np.sum(m) - obj['area']) for m in masks])
    selected_mask = masks[best_mask_idx]

    # Добавляем маску в итоговое изображение с учетом класса
    mask[selected_mask > 0] = class_id

# --- 5. Визуализация ---
# Палитра цветов (фон: черный, класс 1: красный, класс 2: зеленый)
palette = np.array([
    [0, 0, 0],  # Фон (0)
    [0, 0, 255],  # Класс 1 (красный)
    [255, 0, 0],  # Класс 2 (зеленый)
])

# Преобразуем маску в цветное изображение

one_hot_masks = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)

for cls in range(3):
    one_hot_masks[cls] = (mask == cls).astype(np.uint8)

print(one_hot_masks)
colored_mask = palette[mask]

# Отображаем исходное изображение и маску
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Исходное изображение")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_mask)
plt.title("Сегментационная маска")
plt.axis('off')

plt.show()

# --- 6. Сохранение маски ---
cv2.imwrite("segmentation_mask.png", colored_mask)
