import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def imread_cyrillic(path):
    """Чтение изображений с кириллицей в пути"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_cyrillic(path, img):
    """Запись изображений с кириллицей в пути"""
    cv2.imencode('.jpg', img)[1].tofile(path)


def create_tiles(image: np.ndarray, tile_size: int = 512, stride: int = 384) -> List[
    Tuple[np.ndarray, Tuple[int, int]]]:
    h, w = image.shape[:2]
    tiles = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, (y, x)))

    # Крайние тайлы
    if (h - tile_size) % stride != 0:
        for x in range(0, w - tile_size + 1, stride):
            tile = image[h - tile_size:h, x:x + tile_size]
            tiles.append((tile, (h - tile_size, x)))

    if (w - tile_size) % stride != 0:
        for y in range(0, h - tile_size + 1, stride):
            tile = image[y:y + tile_size, w - tile_size:w]
            tiles.append((tile, (y, w - tile_size)))

    return tiles


def process_image(image_path: str, output_dir: str, tile_size: int = 512, stride: int = 384):
    image = imread_cyrillic(image_path)
    if image is None:
        print(f" Не удалось загрузить: {image_path}")
        return []

    print(f"Загружено: {Path(image_path).name}, размер: {image.shape[1]}x{image.shape[0]}")

    tiles = create_tiles(image, tile_size, stride)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tile_paths = []
    for i, (tile, (y, x)) in enumerate(tiles):
        tile_path = str(output / f"tile_{i:04d}_{y}_{x}.jpg")
        imwrite_cyrillic(tile_path, tile)
        tile_paths.append(tile_path)

    return tile_paths


if __name__ == "__main__":
    # Скрипт лежит в "Задача 3...\Задача 3...\tile.py"
    base_path = Path(__file__).resolve().parent
    panorama_dir = base_path / "Панорамы"

    print(f"Ищем в: {panorama_dir}")

    files = list(panorama_dir.glob("*.*"))
    print(f"Найдено файлов: {len(files)}")

    if not files:
        print("Файлы не найдены!")
    else:
        img_path = files[0]
        print(f"Тестируем: {img_path.name}")
        tiles = process_image(str(img_path), "tiles_output")
        print(f"Создано тайлов: {len(tiles)}")