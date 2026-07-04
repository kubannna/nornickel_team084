import os
import re
import sys
import glob
import csv
import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

UM_PER_PX = {"5": 2.0, "10": 1.0, "20": 0.5}
MIN_AREA_PX = 30
MAX_SIDE = None


def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def parse_magnification(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)\s*[xх]", name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def downscale(bgr, max_side=MAX_SIDE):
    if not max_side:
        return bgr, 1.0
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return bgr, scale


def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return gray


def enhance_for_view(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    norm = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def multiotsu_thresholds(gray):
    """3-классовый Оцу, векторизовано. Возвращает (t1, t2)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    P = np.cumsum(prob)
    S = np.cumsum(prob * np.arange(256))
    mu_t = S[-1]
    t = np.arange(256)
    w0 = P[:, None]; m0 = S[:, None]
    wu = P[None, :]; su = S[None, :]
    w1 = wu - w0;    s1 = su - m0
    w2 = 1.0 - wu
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = np.where(w0 > 0, m0 / w0, 0.0)
        mean1 = np.where(w1 > 0, s1 / w1, 0.0)
        mean2 = np.where(w2 > 0, (mu_t - su) / w2, 0.0)
    sigma = (w0 * (mean0 - mu_t) ** 2
             + w1 * (mean1 - mu_t) ** 2
             + w2 * (mean2 - mu_t) ** 2)
    valid = (t[:, None] < t[None, :]) & (w0 > 0) & (w1 > 0) & (w2 > 0)
    sigma = np.where(valid, sigma, -1.0)
    t1, t2 = np.unravel_index(int(np.argmax(sigma)), sigma.shape)
    return int(t1), int(t2)


def remove_small(mask, min_area=MIN_AREA_PX):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8)
    )
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[lbl].astype(np.uint8)) * 255


def segment_sulfides(gray_norm):
    t1, _ = multiotsu_thresholds(gray_norm)
    mask = (gray_norm >= t1).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return remove_small(mask)


def make_overlay(bgr, mask, color=(0, 255, 0), alpha=0.4):
    overlay = bgr.copy()
    overlay[mask > 0] = color
    out = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def _safe_name(s):
    return re.sub(r"[^\w.-]+", "_", s, flags=re.UNICODE)


def run(input_dir, output_dir, max_side=MAX_SIDE, save_overlays=True):
    masks_dir = os.path.join(output_dir, "masks")
    over_dir = os.path.join(output_dir, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    if save_overlays:
        os.makedirs(over_dir, exist_ok=True)

    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
    paths = sorted(set(paths))

    manifest = []
    for idx, path in enumerate(paths):
        bgr = imread_unicode(path)
        if bgr is None:
            print("skip (unreadable):", path)
            continue
        orig_h, orig_w = bgr.shape[:2]
        bgr, scale = downscale(bgr, max_side)
        mag = parse_magnification(path)
        um = UM_PER_PX.get(mag)
        um_eff = (um / scale) if (um and scale) else um
        gray_norm = preprocess(bgr)
        mask = segment_sulfides(gray_norm)

        ore_class = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        base = f"{idx:04d}_{_safe_name(ore_class)}_{_safe_name(stem)}"
        mask_rel = os.path.join("masks", base + "_mask.npy")
        np.save(os.path.join(output_dir, mask_rel), np.packbits(mask > 0))
        if save_overlays:
            imwrite_unicode(os.path.join(over_dir, base + "_overlay.png"),
                            make_overlay(bgr, mask))

        manifest.append({
            "mask_file": mask_rel.replace(os.sep, "/"),
            "mask_h": mask.shape[0],
            "mask_w": mask.shape[1],
            "file": os.path.basename(path),
            "class": ore_class,
            "magnification": mag if mag is not None else "",
            "um_per_px_eff": ("" if um_eff is None else round(um_eff, 6)),
            "scale": round(scale, 6),
            "orig_w": orig_w,
            "orig_h": orig_h,
        })
        print(f"[{idx+1}/{len(paths)}] {ore_class:20s} {stem:25s} "
              f"sulf={mask.mean()/255:.1%}")

    if not manifest:
        print("No images processed. Check input_dir and file paths.")
        return
    man_path = os.path.join(output_dir, "manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nDone: {len(manifest)} масок -> {masks_dir}")
    print(f"Manifest -> {man_path}")


if __name__ == "__main__":
    run(input_dir="dataset/примеры",
        output_dir="out/step1_sulfides",
        max_side=None)
