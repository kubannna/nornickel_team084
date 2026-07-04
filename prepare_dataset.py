# prepare_dataset.py
import os
import shutil
import csv
from pathlib import Path
from dataset_parser import parse_dataset
from talc_processor import process_talc_annotation


def prepare_dataset(base_path: str, output_dir: str):
    output = Path(output_dir)
    img_dir = output / "images"
    mask_dir = output / "masks_talc"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f" Ищем датасет в: {base_path}")
    print(f"   Папка существует: {Path(base_path).exists()}")

    # Проверяем ключевые подпапки
    ch1 = Path(base_path) / "Фото руд по сортам. ч1"
    ch2 = Path(base_path) / "Фото руд по сортам. ч2"
    print(f"   'Фото руд по сортам. ч1' существует: {ch1.exists()}")
    print(f"   'Фото руд по сортам. ч2' существует: {ch2.exists()}")

    if ch1.exists():
        print(f"   Содержимое ч1: {list(ch1.iterdir())}")

    print("\nПарсинг исходного датасета")
    samples = parse_dataset(base_path)
    print(f"   Найдено семплов: {len(samples)}")

    if len(samples) == 0:
        print("Семплы не найдены! Проверь структуру папок.")
        return

    metadata = []
    processed_count = 0

    for idx, sample in enumerate(samples):
        safe_name = f"{idx:04d}_{sample.label}.jpg"
        img_out_path = img_dir / safe_name

        try:
            shutil.copy2(sample.image_path, str(img_out_path))
        except Exception as e:
            print(f"⚠Ошибка копирования {sample.image_path}: {e}")
            continue

        talc_percent = 0.0
        mask_out_path = mask_dir / f"{idx:04d}_{sample.label}.png"

        if sample.mask_path:
            mask = process_talc_annotation(sample.mask_path, str(mask_out_path))
            if mask is not None:
                talc_percent = (mask.sum() / 255) / mask.size * 100
        else:
            import numpy as np
            import cv2
            img = cv2.imdecode(np.fromfile(sample.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                black_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.imencode('.png', black_mask)[1].tofile(str(mask_out_path))

        metadata.append({
            "id": idx,
            "filename": safe_name,
            "original_path": sample.image_path,
            "class": sample.label,
            "has_talc_annotation": 1 if sample.mask_path else 0,
            "talc_percent": round(talc_percent, 2)
        })

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"   Обработано: {processed_count}/{len(samples)}")

    csv_path = output / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "original_path", "class", "has_talc_annotation",
                                               "talc_percent"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nОбработано изображений: {processed_count}")
    print(f"Датасет сохранен в: {output_dir}")
    print(f"Статистика по классам:")
    for cls in ["оталькованная", "рядовая", "труднообогатимая"]:
        count = sum(1 for m in metadata if m["class"] == cls)
        print(f"   {cls}: {count}")


if __name__ == "__main__":
    # Скрипт лежит во вложенной папке, данные на уровень выше
    base_path = Path(__file__).resolve().parent.parent
    output_dir = Path(__file__).resolve().parent / "dataset_ready"

    # Проверяем, есть ли данные по этому пути
    if not (base_path / "Фото руд по сортам. ч1").exists():
        # Если нет — значит parent.parent указал не туда, берем parent
        base_path = Path(__file__).resolve().parent

    print(f" База: {base_path}")
    print(f"   Вывод: {output_dir}\n")

    prepare_dataset(str(base_path), str(output_dir))