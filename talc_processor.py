# talc_processor.py
import cv2
import numpy as np
from pathlib import Path


def detect_blue_lines(image: np.ndarray) -> np.ndarray:
    """Выделяет синие линии разметки из изображения."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 200, 150])
    upper_blue = np.array([135, 255, 255])
    return cv2.inRange(hsv, lower_blue, upper_blue)


def lines_to_filled_mask(line_mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Конвертирует линии в заполненную маску."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(line_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    kernel_small = np.ones((5, 5), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_small)
    return filled_mask


def process_talc_annotation(image_path: str, output_path: str = None) -> np.ndarray:
    """Обрабатывает изображение с синей разметкой талька."""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    line_mask = detect_blue_lines(image)
    filled_mask = lines_to_filled_mask(line_mask, kernel_size=15)

    if output_path:
        cv2.imencode('.png', filled_mask)[1].tofile(output_path)

    return filled_mask


def detect_talc_by_color(image: np.ndarray) -> np.ndarray:
    """
    Детектирует тальк как темные рассеянные области.
    Отфильтровывает крупную матрицу, оставляя только мелкие вкрапления талька.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Темные пиксели (тальк + матрица)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Ищем связные компоненты (пятна) через OpenCV
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    talc_mask = np.zeros_like(dark_mask)

    # Проходим по всем найденным пятнам (пропускаем 0 - это фон)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Тальк - это мелкие рассеянные области
        # Матрица - это огромные пятна
        # Оставляем только пятна размером от 100 до 100500 пикселей.
        if 100 < area < 100500:
            talc_mask[labels == i] = 255

    # Легкая очистка от шума
    kernel = np.ones((3, 3), np.uint8)
    talc_mask = cv2.morphologyEx(talc_mask, cv2.MORPH_OPEN, kernel)

    return talc_mask


if __name__ == "__main__":
    base_path = Path(r"C:\Новая папка\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф")
    mask_folder = base_path / "Фото руд по сортам. ч1" / "Оталькованные руды" / "Области оталькования"

    img_path = next(mask_folder.glob("*.JPG"), None)
    if img_path:
        print(f"Тестируем на: {img_path.name}")
        mask = process_talc_annotation(str(img_path), "test_talc_mask_filled.png")
        if mask is not None:
            talc_percent = (mask.sum() / 255) / mask.size * 100
            print(f"Доля талька: {talc_percent:.2f}%")