# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from augmentations import get_train_augmentations, get_val_augmentations


class OreDataset(Dataset):
    """Датасет для обучения модели сегментации"""

    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform

        # Загружаем метаданные
        metadata = pd.read_csv(self.dataset_dir / "metadata.csv")

        # Сплит 80/20
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            metadata.index, test_size=0.2, random_state=42, stratify=metadata["class"]
        )

        if split == "train":
            self.metadata = metadata.loc[train_idx].reset_index(drop=True)
            self.transform = transform or get_train_augmentations()
        else:
            self.metadata = metadata.loc[val_idx].reset_index(drop=True)
            self.transform = transform or get_val_augmentations()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Загружаем изображение
        img_path = self.dataset_dir / "images" / row["filename"]
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загружаем маску талька
        mask_path = self.dataset_dir / "masks_talc" / row["filename"].replace(".jpg", ".png")
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Ресайз до фиксированного размера (например 512x512)
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # Аугментации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Бинаризуем маску
        mask = (mask > 127).float()

        return {
            "image": image,
            "mask": mask,
            "class": row["class"],
            "talc_percent": row["talc_percent"],
            "filename": row["filename"]
        }


def get_dataloaders(dataset_dir: str, batch_size: int = 8):
    """Создает train/val даталоадеры"""
    train_dataset = OreDataset(dataset_dir, split="train")
    val_dataset = OreDataset(dataset_dir, split="val")

    # Проверяем наличие GPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Тест
    dataset_dir = r"C:\Новая папка\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\dataset_ready"
    train_loader, val_loader = get_dataloaders(dataset_dir, batch_size=4)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    print(f"Classes: {batch['class']}")