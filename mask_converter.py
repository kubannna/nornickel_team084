"""
mask_converter.py

Преобразует изображения с синей ручной разметкой
в бинарные маски для обучения U-Net.

Автор: Team P2
"""

from pathlib import Path
import cv2
import numpy as np


from pathlib import Path

ROOT = Path("Фото руд по сортам. ч1") / "Оталькованные руды" / "Области оталькования"

print(ROOT)
print(ROOT.exists())

IMAGE_FOLDER = ROOT
OUTPUT_FOLDER = ROOT / "Masks"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# Минимальная площадь контура
MIN_AREA = 500


def detect_blue_contours(image):
    """
    Ищет синие линии на изображении.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([95, 80, 40])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def build_binary_mask(blue_mask):
    """
    Заполняет области внутри контура.
    """

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    binary = np.zeros_like(blue_mask)

    valid = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        cv2.drawContours(
            binary,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        valid += 1

    return binary, valid


images = []

for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.bmp", "*.tif"):
    images.extend(sorted(IMAGE_FOLDER.glob(ext)))

print("=" * 70)
print("MASK CONVERTER")
print("=" * 70)
print(f"Найдено изображений: {len(images)}")
print()

images = []

for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.TIF"):
    images.extend(ROOT.glob(ext))

print(f"Найдено файлов: {len(images)}")

for img in images[:5]:
    print(img)


total_masks = 0

for img_path in images:

    print(f"Обработка: {img_path.name}")

    image = cv2.imread(str(img_path))

    if image is None:
        print("Ошибка чтения.")
        continue

    blue = detect_blue_contours(image)

    binary, n = build_binary_mask(blue)

    save_path = OUTPUT_FOLDER / (img_path.stem + ".png")

    cv2.imwrite(str(save_path), binary)

    print(f"Размер изображения : {image.shape[1]} x {image.shape[0]}")
    print(f"Контуров найдено   : {n}")
    print(f"Маска сохранена    : {save_path.name}")

    total_masks += 1

print()
print("готово")
print("-" * 70)
print(f"Создано масок: {total_masks}")
print(f"Папка: {OUTPUT_FOLDER}")