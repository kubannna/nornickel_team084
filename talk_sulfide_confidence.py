from __future__ import annotations
import os
import csv
import glob
import time
import argparse
from typing import Optional
import numpy as np
import cv2
import torch
import train_talc_unet as talc
import segmentation_sulfide as seg

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
            ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP")


def load_model(ckpt: str, device: Optional[str] = None):
    """Грузит обученный U-Net талька. Возвращает (model, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = talc.load_talc_model(ckpt, device)
    model.eval()
    return model, device


def _binary_entropy(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Поточечная энтропия бинарного sigmoid-выхода, нормирована в [0,1]
    (1.0 = p=0.5 максимальная неуверенность, 0.0 = p=0 или p=1)."""
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _imwrite_unicode(path: str, img: np.ndarray) -> None:
    """Запись PNG с поддержкой кириллицы в пути."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


@torch.no_grad()
def compute_talc_ref_stats(ref_dir, n=20):
    """LAB mean/std обучающих talc-кропов для переноса цвета (Reinhard).
    Возвращает (mean, std) np.float32[3] или None."""
    try:
        import load_dataset as ds
        df = ds.load_metadata(ref_dir)
        pos = df[(df["has_talc_annotation"] == 1) & (df["talc_percent"] >= 3.0)].head(n)
        ms, ss = [], []
        for _, row in pos.iterrows():
            im = ds.imread_unicode(row["image_path"], cv2.IMREAD_COLOR)
            if im is None:
                continue
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
            ms.append(lab.mean(0))
            ss.append(lab.std(0))
        if not ms:
            return None
        mean, std = np.mean(ms, axis=0), np.mean(ss, axis=0)
        print(f"[talc-ref] LAB mean={mean.round(1)} std={std.round(1)} (n={len(ms)})")
        return mean, std
    except Exception as e:
        print(f"   [!] talc-ref статистика не посчитана ({e})")
        return None


def _reinhard(bgr, tgt_mean, tgt_std):
    """Перенос цвета: LAB mean/std изображения -> статистике обучающих."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = lab.reshape(-1, 3).mean(0)
    s = lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - m) / s * tgt_std + tgt_mean
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _normalize_for_talc(bgr, mode="none", target_gray=90.0, ref_stats=None):
    """Приводим яркость/контраст/цвет панорамы к виду обучающих.
    mode: none | bright | clahe | grayworld | reinhard.
    reinhard требует ref_stats=(mean,std) из compute_talc_ref_stats."""
    if not mode or mode == "none":
        return bgr
    if mode == "reinhard":
        return bgr if ref_stats is None else _reinhard(bgr, ref_stats[0], ref_stats[1])
    out = bgr
    if mode == "grayworld":
        f = out.astype(np.float32)
        m = f.reshape(-1, 3).mean(0) + 1e-6
        f *= (m.mean() / m)
        out = np.clip(f, 0, 255).astype(np.uint8)
    elif mode == "clahe":
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[..., 0] = cl.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cur = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean() + 1e-6
    f = out.astype(np.float32) * (target_gray / cur)
    return np.clip(f, 0, 255).astype(np.uint8)


def process_panorama(
    image_path: str,
    model,
    device: str,
    tile: int = 512,
    overlap: int = 128,
    talc_prob_thr: float = 0.5,
    batch_tiles: int = 8,
    uncertain_thr: float = 0.5,
    save_dir: Optional[str] = None,
    tile_progress: bool = False,
    talc_downscale: float = 1.0,
    talc_normalize: str = "none",
    talc_ref_stats=None,
) -> Optional[dict]:
    """Полный P1-инференс одной панорамы. Совместим по ключам со старым
    process_panorama коллеги (image_path, talc_percent, full_mask) + доп. выходы P1."""
    t0 = time.time()
    bgr = seg.imread_unicode(image_path)
    if bgr is None:
        print(f"   [!] не удалось прочитать: {image_path}")
        return None

    H0, W0 = bgr.shape[:2]
    if talc_downscale and talc_downscale > 1.0:
        new_w = max(tile, int(round(W0 / talc_downscale)))
        new_h = max(tile, int(round(H0 / talc_downscale)))
        bgr_talc = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        bgr_talc = bgr
    bgr_talc = _normalize_for_talc(bgr_talc, talc_normalize, ref_stats=talc_ref_stats)
    prob_small = talc.predict_prob_map(
        model, bgr_talc, tile=tile, overlap=overlap, device=device,
        batch_tiles=batch_tiles,
    )
    if prob_small.shape[:2] != (H0, W0):
        prob = cv2.resize(prob_small, (W0, H0), interpolation=cv2.INTER_LINEAR)
    else:
        prob = prob_small
    talc_bin = (prob > talc_prob_thr)
    talc_mask = (talc_bin.astype(np.uint8)) * 255
    talc_percent = float(talc_bin.mean() * 100.0)

    ent = _binary_entropy(prob)
    confidence_map = ((1.0 - ent) * 255).astype(np.uint8)
    mean_confidence = float(1.0 - ent.mean())
    uncertain_percent = float((ent > uncertain_thr).mean() * 100.0)

    gray = seg.preprocess(bgr)
    sulfide_mask = seg.segment_sulfides(gray)
    sulfide_percent = float((sulfide_mask > 0).mean() * 100.0)

    seconds = time.time() - t0
    stem = os.path.splitext(os.path.basename(image_path))[0]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _imwrite_unicode(os.path.join(save_dir, f"{stem}_talc_mask.png"), talc_mask)
        _imwrite_unicode(os.path.join(save_dir, f"{stem}_sulfide_mask.png"), sulfide_mask)
        heat = cv2.applyColorMap(255 - confidence_map, cv2.COLORMAP_JET)
        _imwrite_unicode(os.path.join(save_dir, f"{stem}_uncertainty.png"), heat)

    return {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "talc_percent": round(talc_percent, 2),
        "sulfide_percent": round(sulfide_percent, 2),
        "mean_confidence": round(mean_confidence, 4),
        "uncertain_percent": round(uncertain_percent, 2),
        "seconds": round(seconds, 1),
        "talc_mask": talc_mask,
        "sulfide_mask": sulfide_mask,
        "confidence_map": confidence_map,
    }


def batch_process(
    images_dir: str,
    out_dir: str,
    ckpt: str,
    tile: int = 512,
    overlap: int = 128,
    talc_prob_thr: float = 0.5,
    batch_tiles: int = 8,
    limit: int = 0,
    talc_downscale: float = 1.0,
    talc_normalize: str = "none",
    talc_ref_stats=None,
) -> None:
    """Пакетный P1-инференс папки панорам.

    Пишет:
      out_dir/results.csv     — image,talc_percent  (совместимо с RUN_REPORT коллеги)
      out_dir/p1_results.csv  — полные P1-метрики (тальк%, сульфид%, уверенность, время)
      out_dir/masks/          — *_talc_mask.png, *_sulfide_mask.png, *_uncertainty.png
                                (sulfide_mask -> вход для морфологии P2)
    """
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    files = [p for p in sorted(glob.glob(os.path.join(images_dir, "*")))
             if p.endswith(IMG_EXTS)]
    if limit and limit > 0:
        files = files[:limit]
    if not files:
        print(f"[!] не найдено изображений в {images_dir}")
        return

    model, device = load_model(ckpt)
    print(f"[P1] модель загружена на {device}; панорам к обработке: {len(files)}")

    rows = []
    t_all = time.time()
    for i, path in enumerate(files, 1):
        r = process_panorama(
            path, model, device, tile=tile, overlap=overlap,
            talc_prob_thr=talc_prob_thr, batch_tiles=batch_tiles, save_dir=masks_dir,
            talc_downscale=talc_downscale, talc_normalize=talc_normalize,
            talc_ref_stats=talc_ref_stats,
        )
        if r is None:
            continue
        rows.append(r)
        print(f"[{i}/{len(files)}] {r['image']}: "
              f"тальк={r['talc_percent']:.2f}% сульфиды={r['sulfide_percent']:.2f}% "
              f"conf={r['mean_confidence']:.3f} неувер={r['uncertain_percent']:.1f}% "
              f"({r['seconds']:.1f}s)", flush=True)

    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "talc_percent"])
        w.writeheader()
        for r in rows:
            w.writerow({"image": r["image"], "talc_percent": r["talc_percent"]})

    with open(os.path.join(out_dir, "p1_results.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["image", "talc_percent", "sulfide_percent",
                "mean_confidence", "uncertain_percent", "seconds"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    dt = time.time() - t_all
    n = len(rows)
    avg = dt / n if n else 0.0
    slow = [r["image"] for r in rows if r["seconds"] > 300]
    print(f"\n[P1] готово: {n} панорам за {dt:.1f}s (среднее {avg:.1f}s/снимок)")
    print(f"[P1] results.csv -> RUN_REPORT.py; sulfide-маски -> морфология P2 ({masks_dir})")
    if slow:
        print(f"[P1][!] превысили 5 мин: {slow}")


def parse_args():
    ap = argparse.ArgumentParser(description="P1: инференс панорам (тальк U-Net + сульфиды + уверенность)")
    ap.add_argument("--images", default="Панорамы", help="папка с панорамами")
    ap.add_argument("--out", default="p1_out", help="папка вывода")
    ap.add_argument("--ckpt", default="runs_talc/best.pt", help="чекпойнт U-Net талька")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--talc-prob-thr", type=float, default=0.5)
    ap.add_argument("--batch-tiles", type=int, default=8)
    ap.add_argument("--talc-downscale", type=float, default=1.0,
                    help="ужать панораму в N раз перед U-Net талька")
    ap.add_argument("--talc-normalize", default="none",
                    choices=["none", "bright", "clahe", "grayworld", "reinhard"],
                    help="нормализация перед U-Net талька (reinhard = перенос цвета)")
    ap.add_argument("--talc-ref", default="dataset_ready",
                    help="папка обучающих данных для reinhard (LAB-статистика)")
    ap.add_argument("--limit", type=int, default=0, help="0 = все")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    _ref = compute_talc_ref_stats(a.talc_ref) if a.talc_normalize == "reinhard" else None
    batch_process(
        images_dir=a.images, out_dir=a.out, ckpt=a.ckpt,
        tile=a.tile, overlap=a.overlap, talc_prob_thr=a.talc_prob_thr,
        batch_tiles=a.batch_tiles, limit=a.limit, talc_downscale=a.talc_downscale,
        talc_normalize=a.talc_normalize, talc_ref_stats=_ref,
    )
