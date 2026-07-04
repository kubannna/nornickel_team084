from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cv2

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    Dataset = object


CLASS_TO_IDX = {"рядовая": 0, "труднообогатимая": 1, "оталькованная": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

POS_TILE_MIN_FRAC = 0.002


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Чтение изображения с кириллическим путём (Windows-safe)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _parse_class_from_name(filename: str) -> Optional[str]:
    """Фолбэк: класс из имени файла `{id}_{class}.jpg` по ключевым словам."""
    low = os.path.basename(filename).lower()
    if "тальк" in low or "отальк" in low:
        return "оталькованная"
    if "трудн" in low or "тонк" in low:
        return "труднообогатимая"
    if "рядов" in low:
        return "рядовая"
    return None


def _normalize_class(value) -> Optional[str]:
    low = str(value).strip().lower()
    if low in CLASS_TO_IDX:
        return low
    return _parse_class_from_name(low)


def load_metadata(root: str) -> pd.DataFrame:
    """Читает metadata.csv, проставляет абсолютные пути к image/mask и нормализует класс.

    Возвращает DataFrame с колонками:
      id, filename, class, has_talc_annotation, talc_percent, image_path, mask_path
    """
    root = os.path.abspath(root)
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks_talc")
    meta_path = os.path.join(root, "metadata.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"нет metadata.csv в {root}")

    df = pd.read_csv(meta_path)
    if "filename" not in df.columns:
        raise ValueError("в metadata.csv нет колонки 'filename'")

    if "class" in df.columns:
        df["class"] = df["class"].apply(_normalize_class)
    else:
        df["class"] = df["filename"].apply(_parse_class_from_name)

    df["image_path"] = df["filename"].apply(lambda f: os.path.join(images_dir, f))
    df["mask_path"] = df["filename"].apply(
        lambda f: os.path.join(masks_dir, os.path.splitext(f)[0] + ".png")
    )

    if "has_talc_annotation" not in df.columns:
        df["has_talc_annotation"] = df["mask_path"].apply(
            lambda p: 1 if os.path.exists(p) else 0
        )
    if "talc_percent" not in df.columns:
        df["talc_percent"] = 0.0

    df["has_talc_annotation"] = df["has_talc_annotation"].fillna(0).astype(int)
    df["talc_percent"] = pd.to_numeric(df["talc_percent"], errors="coerce").fillna(0.0)

    exists = df["image_path"].apply(os.path.exists)
    missing = int((~exists).sum())
    if missing:
        print(f"[load_metadata] предупреждение: {missing} файлов из CSV не найдено на диске")
    df = df[exists].reset_index(drop=True)
    return df


def make_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify_cols=("class", "has_talc_annotation"),
) -> pd.DataFrame:
    """Стратифицированный сплит НА УРОВНЕ ИЗОБРАЖЕНИЙ по паре
    (класс, есть_ли_разметка_талька) — чтобы 42 маски талька делились
    контролируемо, а не случайно. Добавляет колонку 'split' (train/val/test).
    Тайлинг делается позже -> утечки тайлов между сплитами нет.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"
    cols = list(stratify_cols) if isinstance(stratify_cols, (list, tuple)) else [stratify_cols]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["class"]
    for _, group in df.groupby(cols):
        idx = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return df


def tile_coords(h: int, w: int, tile: int, overlap: int):
    """Координаты левого-верхнего угла тайлов с перекрытием. Последний тайл
    прижимается к краю, чтобы покрыть всё изображение."""
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def _pad_to(img: np.ndarray, tile: int) -> np.ndarray:
    """Дополнить до размера тайла (для маленьких изображений)."""
    h, w = img.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph == 0 and pw == 0:
        return img
    if img.ndim == 3:
        return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


class _ImageCache:
    """Мини-кэш последних открытых файлов, чтобы не перечитывать один шлиф для
    каждого его тайла."""

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._store: dict = {}
        self._order: list = []

    def get(self, path: str, flags: int):
        key = (path, flags)
        if key in self._store:
            return self._store[key]
        img = imread_unicode(path, flags)
        self._store[key] = img
        self._order.append(key)
        if len(self._order) > self.capacity:
            old = self._order.pop(0)
            self._store.pop(old, None)
        return img


def _augment(img: np.ndarray, mask: Optional[np.ndarray], rng: np.random.Generator):
    """Лёгкие аугментации на numpy/cv2 (без albumentations).
    img: HxWx3 uint8, mask: HxW uint8 или None."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
        if mask is not None:
            mask = mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
        if mask is not None:
            mask = np.rot90(mask, k)
    img = img.astype(np.float32)
    if rng.random() < 0.7:
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        img = img * alpha + beta
    if rng.random() < 0.5:
        gamma = float(rng.uniform(0.8, 1.25))
        img = 255.0 * np.clip(img / 255.0, 0, 1) ** gamma
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), (None if mask is None else np.ascontiguousarray(mask))


def _to_chw_tensor(img_uint8: np.ndarray):
    """HxWx3 uint8 (BGR) -> нормализованный CHW float32 tensor (RGB, ImageNet)."""
    rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1)).copy()
    return torch.from_numpy(chw)


@dataclass
class TalcTile:
    row_idx: int
    y: int
    x: int
    positive: bool


class TalcSegmentationDataset(Dataset):
    """Тайлы (изображение + бинарная маска талька) для U-Net.

    Балансировка: все положительные тайлы (из ~42 размеченных шлифов) +
    подвыборка отрицательных до neg_per_pos на каждый положительный.
    На train включены аугментации; на val/test — нет.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        split: str,
        tile: int = 512,
        overlap: int = 64,
        neg_per_pos: Optional[float] = 3.0,
        augment: Optional[bool] = None,
        seed: int = 42,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("torch не установлен")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root = os.path.abspath(root)
        self.tile = tile
        self.overlap = overlap
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.rng = np.random.default_rng(seed)
        self._cache = _ImageCache(capacity=4)
        self.tiles = self._build_index(neg_per_pos)

    def _build_index(self, neg_per_pos: Optional[float]):
        pos: list = []
        neg: list = []
        for ri, row in self.df.iterrows():
            if int(row["has_talc_annotation"]) == 1 and os.path.exists(row["mask_path"]):
                mask = imread_unicode(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                h, w = mask.shape[:2]
                for (y, x) in tile_coords(h, w, self.tile, self.overlap):
                    sub = mask[y:y + self.tile, x:x + self.tile]
                    frac = float((sub > 127).mean()) if sub.size else 0.0
                    t = TalcTile(ri, y, x, frac >= POS_TILE_MIN_FRAC)
                    (pos if t.positive else neg).append(t)
            else:
                img = imread_unicode(row["image_path"], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for (y, x) in tile_coords(h, w, self.tile, self.overlap):
                    neg.append(TalcTile(ri, y, x, False))

        if self.split == "train" and neg_per_pos is not None and len(pos) > 0:
            keep = int(round(len(pos) * neg_per_pos))
            if keep < len(neg):
                sel = self.rng.choice(len(neg), size=keep, replace=False)
                neg = [neg[i] for i in sel]
        tiles = pos + neg
        self.rng.shuffle(tiles)
        print(f"[TalcDataset:{self.split}] тайлов={len(tiles)} "
              f"(pos={len(pos)}, neg={len(neg)}) из {len(self.df)} шлифов")
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i: int):
        t = self.tiles[i]
        row = self.df.iloc[t.row_idx]
        img = self._cache.get(row["image_path"], cv2.IMREAD_COLOR)
        img = _pad_to(img, self.tile)
        img_tile = img[t.y:t.y + self.tile, t.x:t.x + self.tile]

        if int(row["has_talc_annotation"]) == 1 and os.path.exists(row["mask_path"]):
            mask = self._cache.get(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = _pad_to(mask, self.tile)
            mask_tile = (mask[t.y:t.y + self.tile, t.x:t.x + self.tile] > 127).astype(np.uint8)
        else:
            mask_tile = np.zeros((self.tile, self.tile), dtype=np.uint8)

        if self.augment:
            img_tile, mask_tile = _augment(img_tile, mask_tile, self.rng)

        x_tensor = _to_chw_tensor(img_tile)
        y_tensor = torch.from_numpy(mask_tile.astype(np.float32)[None, ...])
        return x_tensor, y_tensor


class OreClassificationDataset(Dataset):
    """Целое изображение (ресайз) + метка класса руды. Для resnet18/34."""

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        split: str,
        img_size: int = 256,
        augment: Optional[bool] = None,
        include_classes: Optional[tuple] = None,
        seed: int = 42,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("torch не установлен")
        sub = df[df["split"] == split].copy()
        sub = sub[sub["class"].notna()]
        if include_classes is not None:
            sub = sub[sub["class"].isin(include_classes)]
        self.df = sub.reset_index(drop=True)
        self.root = os.path.abspath(root)
        self.img_size = img_size
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.rng = np.random.default_rng(seed)
        print(f"[OreClsDataset:{split}] изображений={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = imread_unicode(row["image_path"], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            img, _ = _augment(img, None, self.rng)
        x_tensor = _to_chw_tensor(img)
        y = CLASS_TO_IDX[row["class"]]
        return x_tensor, y


def build_talc_loaders(df, root, tile=512, overlap=64, batch_size=8,
                       neg_per_pos=3.0, num_workers=0, seed=42):
    tr = TalcSegmentationDataset(df, root, "train", tile, overlap, neg_per_pos, seed=seed)
    va = TalcSegmentationDataset(df, root, "val", tile, overlap, neg_per_pos=None,
                                 augment=False, seed=seed)
    te = TalcSegmentationDataset(df, root, "test", tile, overlap, neg_per_pos=None,
                                 augment=False, seed=seed)
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, drop_last=False)
    return dl(tr, True), dl(va, False), dl(te, False)


def class_weights(df, split="train", include_classes=None):
    """Веса классов (обратные частоте) для CrossEntropyLoss."""
    sub = df[(df["split"] == split) & df["class"].notna()]
    if include_classes is not None:
        sub = sub[sub["class"].isin(include_classes)]
    counts = sub["class"].value_counts()
    classes = list(include_classes) if include_classes else list(CLASS_TO_IDX.keys())
    w = np.array([1.0 / max(1, int(counts.get(c, 0))) for c in classes], dtype=np.float32)
    w = w / w.sum() * len(classes)
    return classes, w


def build_cls_loaders(df, root, img_size=256, batch_size=16,
                      include_classes=None, num_workers=0, seed=42):
    tr = OreClassificationDataset(df, root, "train", img_size, include_classes=include_classes, seed=seed)
    va = OreClassificationDataset(df, root, "val", img_size, augment=False, include_classes=include_classes, seed=seed)
    te = OreClassificationDataset(df, root, "test", img_size, augment=False, include_classes=include_classes, seed=seed)
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, drop_last=False)
    return dl(tr, True), dl(va, False), dl(te, False), CLASS_TO_IDX


def _selftest(root: str):
    print(f"== Самопроверка загрузчика на {root} ==")
    df = load_metadata(root)
    print(f"Всего изображений: {len(df)}")
    print("По классам:\n", df["class"].value_counts(dropna=False).to_string())
    n_ann = int((df["has_talc_annotation"] == 1).sum())
    print(f"С разметкой талька: {n_ann}")
    if n_ann:
        print(f"Средний talc_percent (размеченные): "
              f"{df.loc[df['has_talc_annotation']==1,'talc_percent'].mean():.2f}")

    df = make_splits(df)
    print("Сплит:\n", df.groupby(["split", "class"]).size().to_string())
    print("Разметка талька по сплиту:\n",
          df[df["has_talc_annotation"] == 1].groupby("split").size().to_string())

    if _HAS_TORCH:
        tr, va, te = build_talc_loaders(df, root, batch_size=4, num_workers=0)
        xb, yb = next(iter(tr))
        print(f"Тальк batch: X={tuple(xb.shape)} Y={tuple(yb.shape)} "
              f"pos_pix={float(yb.mean()):.4f}")
        ctr, cva, cte, c2i = build_cls_loaders(df, root, batch_size=4, num_workers=0)
        cx, cy = next(iter(ctr))
        print(f"Класс batch: X={tuple(cx.shape)} y={cy.tolist()} map={c2i}")
        classes, w = class_weights(df)
        print(f"Веса классов: {dict(zip(classes, [round(float(x),3) for x in w]))}")
    else:
        print("torch не найден — проверил только метаданные/сплит. Поставь torch, чтобы увидеть тайлы.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Загрузчик dataset_ready (Шаг 4)")
    ap.add_argument("--root", default="dataset_ready", help="папка с images/ masks_talc/ metadata.csv")
    args = ap.parse_args()
    _selftest(args.root)
