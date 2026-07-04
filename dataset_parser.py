import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Sample:
    image_path: str
    mask_path: Optional[str] = None  # маска талька
    label: Optional[str] = None  # оталькованная/рядовая/труднообогатимая


def parse_dataset(base_path: str) -> List[Sample]:
    base = Path(base_path)
    samples = []

    # Ч1: Оталькованные руды + вложенная папка Области оталькования
    talc_folder = base / "Фото руд по сортам. ч1" / "Оталькованные руды"
    if talc_folder.exists():
        mask_folder = talc_folder / "Области оталькования"
        for img_file in talc_folder.glob("*.JPG"):
            mask_file = mask_folder / img_file.name if mask_folder.exists() else None
            samples.append(Sample(
                image_path=str(img_file),
                mask_path=str(mask_file) if mask_file and mask_file.exists() else None,
                label="оталькованная"
            ))

    # Ч1: Рядовые руды
    normal_folder = base / "Фото руд по сортам. ч1" / "Рядовые руды"
    if normal_folder.exists():
        for img_file in normal_folder.glob("*.JPG"):
            samples.append(Sample(image_path=str(img_file), label="рядовая"))

    # Ч1: Труднообогатимые руды
    fine_folder = base / "Фото руд по сортам. ч1" / "Труднообогатимые руды"
    if fine_folder.exists():
        for img_file in fine_folder.glob("*.JPG"):
            samples.append(Sample(image_path=str(img_file), label="труднообогатимая"))

    # Ч2: оталькованные
    talc2 = base / "Фото руд по сортам. ч2" / "оталькованные"
    if talc2.exists():
        for img_file in talc2.glob("*.jpg"):
            samples.append(Sample(image_path=str(img_file), label="оталькованная"))

    # Ч2: рядовые
    normal2 = base / "Фото руд по сортам. ч2" / "рядовые"
    if normal2.exists():
        for img_file in normal2.glob("*.jpg"):
            samples.append(Sample(image_path=str(img_file), label="рядовая"))

    # Ч2: тонкие (труднообогатимые)
    fine2 = base / "Фото руд по сортам. ч2" / "тонкие"
    if fine2.exists():
        for img_file in fine2.glob("*.jpg"):
            samples.append(Sample(image_path=str(img_file), label="труднообогатимая"))

    return samples


if __name__ == "__main__":
    base_path = r"C:\Новая папка\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф"
    samples = parse_dataset(base_path)

    print(f"Всего семплов: {len(samples)}")
    print(f"Оталькованные: {sum(1 for s in samples if s.label == 'оталькованная')}")
    print(f"Рядовые: {sum(1 for s in samples if s.label == 'рядовая')}")
    print(f"Труднообогатимые: {sum(1 for s in samples if s.label == 'труднообогатимая')}")
    print(f"С масками талька: {sum(1 for s in samples if s.mask_path)}")

    for s in samples[:5]:
        print(f"\n{os.path.basename(s.image_path)}: {s.label}")
        if s.mask_path:
            print(f"  Маска: {os.path.basename(s.mask_path)}")