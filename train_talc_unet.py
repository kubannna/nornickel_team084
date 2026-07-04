from __future__ import annotations
import os
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import load_dataset as ds

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except Exception:
    _HAS_SMP = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def _progress(iterable, desc: str, total: Optional[int] = None):
    """Обёртка tqdm с фолбэком: если tqdm нет — возвращает исходный итератор."""
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=100, leave=False)
    return iterable


def build_model(encoder: str = "efficientnet-b0") -> nn.Module:
    if not _HAS_SMP:
        raise RuntimeError(
            "нет segmentation-models-pytorch. Установи: "
            "pip install segmentation-models-pytorch"
        )
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    num = 2.0 * (prob * target).sum(dims) + eps
    den = prob.sum(dims) + target.sum(dims) + eps
    return (1.0 - num / den).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: Optional[float] = None):
        super().__init__()
        self.bce_weight = bce_weight
        pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, target):
        return dice_loss(logits, target) + self.bce_weight * self.bce(logits, target)


@torch.no_grad()
def dice_iou(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = ((2 * inter + 1) / (union + 1)).mean().item()
    iou = ((inter + 1) / (union - inter + 1)).mean().item()
    return dice, iou


def subsample_val_tiles(val_ds, neg_per_pos: float, seed: int = 42):
    """Оставляет ВСЕ положительные тайлы + подвыборку негативов (neg_per_pos на 1 полож.).
    Ускоряет тайловый Dice/IoU-монитор и делает его сбалансированным.
    На метрику +-3% НЕ влияет (та считается полнокадрово отдельно)."""
    if neg_per_pos is None:
        return val_ds
    pos = [t for t in val_ds.tiles if t.positive]
    neg = [t for t in val_ds.tiles if not t.positive]
    if len(pos) == 0:
        return val_ds
    keep = int(round(len(pos) * neg_per_pos))
    rng = np.random.default_rng(seed)
    if keep < len(neg):
        sel = rng.choice(len(neg), size=keep, replace=False)
        neg = [neg[i] for i in sel]
    val_ds.tiles = pos + neg
    rng.shuffle(val_ds.tiles)
    print(f"[val-monitor] подвыборка: тайлов={len(val_ds.tiles)} (pos={len(pos)}, neg={len(neg)})")
    return val_ds


@torch.no_grad()
def predict_prob_map(
    model: nn.Module,
    image_bgr: np.ndarray,
    tile: int = 512,
    overlap: int = 128,
    device: str = "cuda",
    batch_tiles: int = 8,
) -> np.ndarray:
    """Нарезает изображение на тайлы, предсказывает вероятность талька,
    сшивает обратно (усреднение перекрытий). Возвращает prob-карту HxW в [0,1]."""
    model.eval()
    h0, w0 = image_bgr.shape[:2]
    img = ds._pad_to(image_bgr, tile)
    H, W = img.shape[:2]
    prob_sum = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    coords = ds.tile_coords(H, W, tile, overlap)
    batch, positions = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        for (yy, xx), p in zip(positions, probs):
            prob_sum[yy:yy + tile, xx:xx + tile] += p
            cnt[yy:yy + tile, xx:xx + tile] += 1.0
        batch.clear()
        positions.clear()

    for (y, x) in coords:
        t = img[y:y + tile, x:x + tile]
        batch.append(ds._to_chw_tensor(t))
        positions.append((y, x))
        if len(batch) >= batch_tiles:
            flush()
    flush()

    cnt[cnt == 0] = 1.0
    prob = prob_sum / cnt
    return prob[:h0, :w0]


@torch.no_grad()
def predict_talc_fraction(
    model: nn.Module,
    image_path: str,
    tile: int = 512,
    overlap: int = 128,
    device: str = "cuda",
    thr: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """Доля талька в % по целому снимку + prob-карта."""
    img = ds.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0, np.zeros((1, 1), dtype=np.float32)
    prob = predict_prob_map(model, img, tile, overlap, device)
    pct = float((prob > thr).mean() * 100.0)
    return pct, prob


def fallback_talc_fraction(image_path: str, sat_max: int = 60, val_min: int = 110) -> float:
    """Наивный детектор талька без обучения: тальк на шлифах — светлые
    серовато-белые области (низкая насыщенность + средняя/высокая яркость)."""
    img = ds.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (s <= sat_max) & (v >= val_min)
    mask = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return float(mask.mean() * 100.0)


@torch.no_grad()
def eval_talc_percent(
    model: nn.Module,
    df: pd.DataFrame,
    root: str,
    split: str,
    tile: int,
    overlap: int,
    device: str,
    tol: float = 3.0,
):
    """Метрика +-3% по РАЗМЕЧЕННЫМ снимкам split. Возвращает (mae, within_tol, n)."""
    sub = df[(df["split"] == split) & (df["has_talc_annotation"] == 1)]
    if len(sub) == 0:
        return float("nan"), float("nan"), 0
    errs, ok = [], 0
    for _, row in sub.iterrows():
        pred_pct, _ = predict_talc_fraction(model, row["image_path"], tile, overlap, device)
        mask = ds.imread_unicode(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            true_pct = float((mask > 127).mean() * 100.0)
        else:
            true_pct = float(row["talc_percent"])
        err = abs(pred_pct - true_pct)
        errs.append(err)
        ok += int(err <= tol)
    mae = float(np.mean(errs))
    within = ok / len(sub)
    return mae, within, len(sub)


@torch.no_grad()
def eval_unannotated_leak(model, df, root, split, tile, overlap, device, limit=40):
    """Санити-чек: средний предсказанный тальк на НЕразмеченных (ждём ~0)."""
    sub = df[(df["split"] == split) & (df["has_talc_annotation"] == 0)]
    if len(sub) == 0:
        return float("nan")
    sub = sub.sample(min(limit, len(sub)), random_state=0)
    vals = [predict_talc_fraction(model, r["image_path"], tile, overlap, device)[0]
            for _, r in sub.iterrows()]
    return float(np.mean(vals))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[!] CUDA не найдена — обучение на CPU будет медленным.")
    if not _HAS_TQDM:
        print("[i] tqdm не установлен — прогресс-бара не будет (pip install tqdm).")
    os.makedirs(args.out, exist_ok=True)

    df = ds.make_splits(ds.load_metadata(args.root), seed=args.seed)

    train_ds = ds.TalcSegmentationDataset(
        df, args.root, "train", tile=args.tile, overlap=args.overlap,
        neg_per_pos=args.neg_per_pos, seed=args.seed,
    )
    val_ds = ds.TalcSegmentationDataset(
        df, args.root, "val", tile=args.tile, overlap=args.overlap,
        neg_per_pos=None, augment=False, seed=args.seed,
    )
    val_ds = subsample_val_tiles(val_ds, args.val_neg_per_pos, seed=args.seed)

    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh,
                                  num_workers=args.num_workers, drop_last=False)
    tr, va = dl(train_ds, True), dl(val_ds, False)

    model = build_model(args.encoder).to(device)
    criterion = DiceBCELoss(bce_weight=0.5, pos_weight=args.pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = float("inf")
    best_path = os.path.join(args.out, "best.pt")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        tbar = _progress(tr, f"E{epoch:02d} train", total=len(tr))
        for x, y in tbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            nb += 1
            if _HAS_TQDM:
                tbar.set_postfix(loss=f"{run_loss / max(1, nb):.4f}")
        sched.step()

        model.eval()
        dices, ious, vloss = [], [], 0.0
        vbar = _progress(va, f"E{epoch:02d} val", total=len(va))
        with torch.no_grad():
            for x, y in vbar:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    vloss += criterion(logits, y).item()
                d, i = dice_iou(logits, y)
                dices.append(d)
                ious.append(i)
                if _HAS_TQDM:
                    vbar.set_postfix(dice=f"{np.mean(dices):.3f}")
        vdice = float(np.mean(dices)) if dices else float("nan")
        viou = float(np.mean(ious)) if ious else float("nan")

        mae_bar = _progress(
            df[(df["split"] == "val") & (df["has_talc_annotation"] == 1)].iterrows(),
            f"E{epoch:02d} +-3%",
        )
        if _HAS_TQDM:
            for _ in mae_bar:
                pass
        mae, within, n = eval_talc_percent(
            model, df, args.root, "val", args.tile, args.overlap, device, tol=args.tol
        )

        dt = time.time() - t0
        print(f"[E{epoch:02d}/{args.epochs}] "
              f"train_loss={run_loss/max(1,nb):.4f} val_loss={vloss/max(1,len(va)):.4f} "
              f"Dice={vdice:.3f} IoU={viou:.3f} | "
              f"talc_MAE={mae:.2f}% within+-{args.tol:.0f}%={within*100:.0f}% (n={n}) "
              f"[{dt:.0f}s]")

        if mae < best_metric - 1e-6:
            best_metric = mae
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "encoder": args.encoder,
                "tile": args.tile,
                "overlap": args.overlap,
                "val_talc_mae": mae,
                "val_within_tol": within,
            }, best_path)
            print(f"    -> сохранён best.pt (talc_MAE={mae:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"    -> Early stop: val talc-MAE не улучшался {args.patience} эпох подряд")
                break

    print(f"\nЛучший val talc-MAE: {best_metric:.2f}% -> {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    mae_t, within_t, n_t = eval_talc_percent(
        model, df, args.root, "test", args.tile, args.overlap, device, tol=args.tol
    )
    leak = eval_unannotated_leak(model, df, args.root, "val", args.tile, args.overlap, device)
    print(f"[TEST] talc_MAE={mae_t:.2f}% within+-{args.tol:.0f}%={within_t*100:.0f}% (n={n_t})")
    print(f"[Санити] средний предсказанный тальк на НЕразмеченных val: {leak:.2f}% (ждём низко)")

    sub = df[(df["split"] == "test") & (df["has_talc_annotation"] == 1)]
    if len(sub):
        fb_err = []
        for _, row in sub.iterrows():
            mask = ds.imread_unicode(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            true_pct = float((mask > 127).mean() * 100.0) if mask is not None else float(row["talc_percent"])
            fb_err.append(abs(fallback_talc_fraction(row["image_path"]) - true_pct))
        print(f"[FALLBACK] talc_MAE={np.mean(fb_err):.2f}% (наивный цвет/яркость, для сравнения)")

    _save_previews(model, df, args, device)


@torch.no_grad()
def _save_previews(model, df, args, device, k=6):
    sub = df[(df["split"] == "val") & (df["has_talc_annotation"] == 1)].head(k)
    out = os.path.join(args.out, "previews")
    os.makedirs(out, exist_ok=True)
    for _, row in sub.iterrows():
        img = ds.imread_unicode(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        prob = predict_prob_map(model, img, args.tile, args.overlap, device)
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        name = os.path.splitext(os.path.basename(row["image_path"]))[0]
        ds_path = os.path.join(out, f"{name}_overlay.png")
        cv2.imencode(".png", overlay)[1].tofile(ds_path)
    print(f"[previews] сохранены в {out}")


def load_talc_model(ckpt_path: str, device: str = "cuda") -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt.get("encoder", "efficientnet-b0")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def parse_args():
    ap = argparse.ArgumentParser(description="Обучение U-Net сегментации талька (Шаг 4)")
    ap.add_argument("--root", default="dataset_ready")
    ap.add_argument("--out", default="runs_talc")
    ap.add_argument("--encoder", default="efficientnet-b0",
                    help="efficientnet-b0 / resnet34 / mobilenet_v2 ...")
    ap.add_argument("--epochs", type=int, default=40, help="максимум эпох (early stop обычно раньше)")
    ap.add_argument("--patience", type=int, default=8,
                    help="early stop: сколько эпох без улучшения val talc-MAE терпеть")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--neg-per-pos", type=float, default=3.0, help="баланс негативов на train")
    ap.add_argument("--val-neg-per-pos", type=float, default=3.0,
                    help="подвыборка негативов для быстрого Dice-монитора (None = все тайлы)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="pos_weight для BCE (напр. 5-10 при сильном дисбалансе)")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
