# -*- coding: utf-8 -*-
"""
active_learning.py — ручная коррекция масок + накопление обучающих примеров.

Две функции:
  - reclassify_from_masks(...)  — пересчёт класса по ПРАВЛЕНЫМ маскам
                                  без запуска нейросетей (доли секунды).
  - save_correction(...)        — сохраняет снимок + правленые маски в пул
                                  коррекций в формате, который читает load_dataset.

Пул коррекций складывается как готовый датасет-root (images/ + masks_talc/ +
metadata.csv), поэтому его можно сразу подать в дообучение:

    python train_talc_unet.py --root dataset_ready/corrections \\
           --init-from runs_talc/best.pt --freeze-encoder --lr 1e-4 --out runs_talc_ft
    python ore_classifier.py  --root dataset_ready/corrections \\
           --init-from runs_cls/best.pt  --freeze-backbone --lr 3e-5 --out runs_cls_ft
"""
import os
import csv
import shutil
import numpy as np
import cv2

from morphology import analyze_sulfide_objects, compute_intergrowth_ratio
from domain_rules import classify_ore


def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Любую маску -> uint8 {0,255}."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return ((m > 127).astype(np.uint8)) * 255


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Запись PNG с поддержкой не-ASCII путей (как imread_unicode в load_dataset)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def reclassify_from_masks(
    talc_mask: np.ndarray,
    sulfide_mask: np.ndarray,
    um_per_px: float | None = None,
) -> dict:
    """Пересчёт класса по правленым маскам (без нейросетей).

    talc_percent — доля площади талька; normal/fine — из морфологии сульфидов.
    Совместимо с domain_rules.classify_ore.
    """
    talc_bin = _to_binary(talc_mask)
    sulf_bin = _to_binary(sulfide_mask)

    talc_percent = float((talc_bin > 127).mean() * 100.0)
    sulfide_percent = float((sulf_bin > 127).mean() * 100.0)

    objects = analyze_sulfide_objects(sulf_bin)
    ratio = compute_intergrowth_ratio(objects)

    dr = classify_ore(talc_percent, ratio["normal_percent"], ratio["fine_percent"])
    return {
        "class": dr["class"],
        "class_source": "manual_correction",
        "description": dr["description"],
        "talc_percent": round(talc_percent, 2),
        "sulfide_percent": round(sulfide_percent, 2),
        "normal_percent": round(ratio["normal_percent"], 2),
        "fine_percent": round(ratio["fine_percent"], 2),
        "n_objects": len(objects),
    }


def save_correction(
    image_path: str,
    talc_mask: np.ndarray,
    sulfide_mask: np.ndarray,
    ore_class: str | None = None,
    out_root: str = "dataset_ready/corrections",
) -> dict:
    """Сохраняет исправленный пример в пул для дообучения.

    Структура out_root (совместима с load_dataset.load_metadata):
        out_root/images/<name>
        out_root/masks_talc/<stem>.png       # маска талька (для U-Net)
        out_root/masks_sulfide/<stem>.png    # маска сульфидов (на будущее)
        out_root/metadata.csv                # filename, class, has_talc_annotation, talc_percent
    """
    images_dir = os.path.join(out_root, "images")
    talc_dir = os.path.join(out_root, "masks_talc")
    sulf_dir = os.path.join(out_root, "masks_sulfide")
    for d in (images_dir, talc_dir, sulf_dir):
        os.makedirs(d, exist_ok=True)

    name = os.path.basename(image_path)
    stem = os.path.splitext(name)[0]

    # исходный снимок
    dst_img = os.path.join(images_dir, name)
    try:
        if os.path.abspath(image_path) != os.path.abspath(dst_img):
            shutil.copyfile(image_path, dst_img)
    except OSError:
        pass

    # маски
    talc_bin = _to_binary(talc_mask)
    sulf_bin = _to_binary(sulfide_mask)
    _imwrite_unicode(os.path.join(talc_dir, stem + ".png"), talc_bin)
    _imwrite_unicode(os.path.join(sulf_dir, stem + ".png"), sulf_bin)

    talc_percent = float((talc_bin > 127).mean() * 100.0)

    # дозапись в metadata.csv
    meta_path = os.path.join(out_root, "metadata.csv")
    header = ["filename", "class", "has_talc_annotation", "talc_percent"]
    new_file = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([name, ore_class or "", 1, round(talc_percent, 3)])

    return {
        "saved_image": dst_img,
        "saved_talc_mask": os.path.join(talc_dir, stem + ".png"),
        "saved_sulfide_mask": os.path.join(sulf_dir, stem + ".png"),
        "metadata": meta_path,
        "class": ore_class,
        "talc_percent": round(talc_percent, 3),
    }
