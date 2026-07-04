# inference_pipeline.py
import cv2
import numpy as np
from pathlib import Path
from tile import create_tiles
from talc_processor import detect_talc_by_color
import csv


def process_panorama(image_path: str, tile_size: int = 512, stride: int = 384):
    """
    Обрабатывает одну панораму: тайлинг -> детект талька -> анализ.
    """
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]

    # Нарезаем на тайлы
    tiles = create_tiles(image, tile_size, stride)

    # Детектим тальк на каждом тайле
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for tile, (y, x) in tiles:
        talc_mask = detect_talc_by_color(tile)
        # Накладываем маску на полное изображение
        full_mask[y:y + tile_size, x:x + tile_size] = np.maximum(
            full_mask[y:y + tile_size, x:x + tile_size],
            talc_mask
        )

    # Считаем процент талька
    talc_pixels = np.sum(full_mask > 0)
    total_pixels = full_mask.size
    talc_percent = (talc_pixels / total_pixels) * 100

    return {
        "image_path": image_path,
        "talc_percent": talc_percent,
        "full_mask": full_mask
    }


def batch_process(panorama_dir: str, output_dir: str):
    """Пакетная обработка всех панорам"""
    panorama_path = Path(panorama_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for img_file in panorama_path.glob("*.*"):
        print(f"Обрабатываем: {img_file.name}")
        result = process_panorama(str(img_file))
        if result:
            results.append(result)

            # Сохраняем маску
            mask_out = output_path / f"{img_file.stem}_talc_mask.png"
            cv2.imencode('.png', result["full_mask"])[1].tofile(str(mask_out))

            print(f"   Тальк: {result['talc_percent']:.2f}%")

    # Сохраняем отчет
    csv_path = output_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "talc_percent"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image_path"],
                "talc_percent": round(r["talc_percent"], 2)
            })

    print(f"\nОбработано панорам: {len(results)}")
    print(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent
    panorama_dir = base_path / "Панорамы"
    output_dir = Path(__file__).resolve().parent / "inference_results"

    batch_process(str(panorama_dir), str(output_dir))
