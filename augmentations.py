# augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_augmentations():
    """Аугментации для обучения"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),

        # Цветовые вариации (имитация разного освещения микроскопа)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

        # Шум и артефакты (обновленный API)
        A.GaussNoise(std_limit=(10, 50), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Имитация царапин и загрязнений шлифа (обновленный API)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            p=0.2
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_augmentations():
    """Аугментации только для валидации (без случайных трансформаций)"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_augmentations():
    """Для инференса на панорамах"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])