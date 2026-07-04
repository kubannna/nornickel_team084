from __future__ import annotations

import os
import json
import time
import argparse
from typing import Optional, List

import numpy as np
import cv2
import torch
import torch.nn as nn

import load_dataset as ds


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision import models
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    try:
        model = models.resnet18(weights=weights)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)
    if weights is None and pretrained:
        print("[!] предобученные веса недоступны — учим с нуля (хуже на малых данных)")
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model


def macro_f1(y_true: List[int], y_pred: List[int], idxs: List[int]):
    """macro-F1 + per-class (только по классам из idxs)."""
    per = {}
    for c in idxs:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = (prec, rec, f1, tp, fp, fn)
    mf1 = sum(v[2] for v in per.values()) / len(per) if per else 0.0
    return mf1, per


def confusion(y_true, y_pred, idxs):
    m = {(t, p): 0 for t in idxs for p in idxs}
    for t, p in zip(y_true, y_pred):
        if (t, p) in m:
            m[(t, p)] += 1
    return m


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy().tolist()
        ps.extend(pred)
        ys.extend([int(v) for v in y])
    return ys, ps


def _report(ys, ps, idxs, title=""):
    mf1, per = macro_f1(ys, ps, idxs)
    acc = sum(1 for t, p in zip(ys, ps) if t == p) / len(ys) if ys else 0.0
    print(f"\n=== {title} === (n={len(ys)})")
    print(f"accuracy = {acc:.3f}")
    for c in idxs:
        prec, rec, f1, tp, fp, fn = per[c]
        print(f"{ds.IDX_TO_CLASS[c]:<18} P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")
    print(f"macro-F1 = {mf1:.3f}")
    m = confusion(ys, ps, idxs)
    header = "        " + "".join(f"{ds.IDX_TO_CLASS[c][:8]:>10}" for c in idxs)
    print("Матрица (стр=истина, столб=прогноз):")
    print(header)
    for t in idxs:
        row = "".join(f"{m[(t, p)]:>10}" for p in idxs)
        print(f"{ds.IDX_TO_CLASS[t][:8]:>8}{row}")
    return mf1, acc


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inc = tuple(args.classes)
    idxs = sorted(ds.CLASS_TO_IDX[c] for c in inc)
    num_classes = max(idxs) + 1

    df = ds.load_metadata(args.root)
    df = ds.make_splits(df, seed=args.seed)
    tr, va, te, _ = ds.build_cls_loaders(
        df, args.root, img_size=args.img_size, batch_size=args.batch_size,
        include_classes=inc, num_workers=args.workers, seed=args.seed,
    )

    classes, w = ds.class_weights(df, "train", include_classes=inc)
    w_by_label = np.ones(num_classes, dtype=np.float32)
    for cname, wv in zip(classes, w):
        w_by_label[ds.CLASS_TO_IDX[cname]] = wv
    weight = torch.tensor(w_by_label, device=device)
    print(f"[cls] классы={inc} веса={dict(zip(classes, [round(float(x),3) for x in w]))}")

    model = build_resnet18(num_classes, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "best.pt")
    best_f1, best_ep, patience = -1.0, -1, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, seen = 0.0, 0
        for x, y in tr:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * x.size(0)
            seen += x.size(0)
        sched.step()
        train_loss = tot / max(1, seen)

        ys, ps = evaluate(model, va, device)
        vf1, per = macro_f1(ys, ps, idxs)
        vacc = sum(1 for t, p in zip(ys, ps) if t == p) / len(ys) if ys else 0.0
        f1_str = " ".join(f"{ds.IDX_TO_CLASS[c][:6]}={per[c][2]:.2f}" for c in idxs)
        print(f"[E{ep:02d}/{args.epochs}] train_loss={train_loss:.4f} "
              f"val_acc={vacc:.3f} val_macroF1={vf1:.3f} ({f1_str}) [{time.time()-t0:.0f}s]", flush=True)

        if vf1 > best_f1:
            best_f1, best_ep, patience = vf1, ep, 0
            torch.save({
                "model": model.state_dict(),
                "arch": "resnet18",
                "num_classes": num_classes,
                "classes": [ds.IDX_TO_CLASS[i] for i in range(num_classes)],
                "included_idxs": idxs,
                "img_size": args.img_size,
            }, ckpt_path)
            print(f"  -> сохранён best.pt (val macro-F1={best_f1:.3f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  -> Early stop: val macro-F1 не рос {args.patience} эпох")
                break

    print(f"\nЛучший val macro-F1: {best_f1:.3f} (E{best_ep}) -> {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    ys, ps = evaluate(model, te, device)
    tf1, tacc = _report(ys, ps, idxs, title="TEST (best.pt)")
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_macro_f1": best_f1, "best_epoch": best_ep,
                   "test_macro_f1": tf1, "test_acc": tacc,
                   "classes": list(inc), "img_size": args.img_size}, f, ensure_ascii=False, indent=2)


def load_classifier(ckpt: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    model = build_resnet18(state["num_classes"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device, state


@torch.no_grad()
def predict_class(model, image_path: str, img_size: int, device: str, included_idxs: List[int]):
    """Возвращает (class_name, prob) только среди included_idxs (рядовая/труднооб)."""
    img = ds.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = ds._to_chw_tensor(img).unsqueeze(0).to(device)
    logits = model(x)[0]
    prob = torch.softmax(logits, dim=0).cpu().numpy()
    best_c = max(included_idxs, key=lambda c: prob[c])
    return ds.IDX_TO_CLASS[best_c], float(prob[best_c])


def detect_crop_tile(root: str) -> int:
    """Медианный размер обучающих кропов -> реком. размер тайла классификатора."""
    try:
        df = ds.load_metadata(root)
    except Exception as e:
        print(f"[cls-tile] не удалось прочитать {root}: {e} -> тайл=1988")
        return 1988
    hs, ws = [], []
    for _cls, grp in df.groupby("class"):
        for p in grp.sample(min(len(grp), 15), random_state=42)["image_path"]:
            im = ds.imread_unicode(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            hs.append(im.shape[0]); ws.append(im.shape[1])
    if not hs:
        return 1988
    mh, mw = int(np.median(hs)), int(np.median(ws))
    return max(128, int(round((mh + mw) / 2)))


def predict_class_tiled(model, image_path: str, img_size: int, device: str,
                        included_idxs: List[int], tile: int = 1988,
                        maxside: int = 12000, min_gray: float = 12.0, batch: int = 32):
    """Классификация панорамы ПО ТАЙЛАМ масштаба обучения + агрегация.

    Режет панораму на неперекрывающиеся тайлы ~размера обучающих кропов,
    классифицирует каждый, усредняет вероятности по included-классам.
    Возвращает (class_name, mean_conf, vote_info) или (None, 0.0, None).
    vote_info: {класс: число_тайлов, "_n_tiles": N}.
    """
    img = ds.imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0, None
    h, w = img.shape[:2]
    sc = min(1.0, maxside / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)))
        h, w = img.shape[:2]
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            t = img[y:min(y + tile, h), x:min(x + tile, w)]
            if t.shape[0] < tile * 0.5 or t.shape[1] < tile * 0.5:
                continue
            if cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).mean() < min_gray:
                continue
            tiles.append(t)
    if not tiles:
        tiles = [img]
    probs = []
    for i in range(0, len(tiles), batch):
        xs = [ds._to_chw_tensor(cv2.resize(t, (img_size, img_size), interpolation=cv2.INTER_AREA))
              for t in tiles[i:i + batch]]
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    probs = np.concatenate(probs, 0)
    sub = probs[:, included_idxs]
    mean_prob = sub.mean(0)
    best_k = int(mean_prob.argmax())
    votes = sub.argmax(1)
    vote_info = {ds.IDX_TO_CLASS[included_idxs[k]]: int((votes == k).sum())
                 for k in range(len(included_idxs))}
    vote_info["_n_tiles"] = len(tiles)
    return ds.IDX_TO_CLASS[included_idxs[best_k]], float(mean_prob[best_k]), vote_info


def parse_args():
    ap = argparse.ArgumentParser(description="P1: resnet18 рядовая/труднооб классификатор")
    ap.add_argument("--root", default="dataset_ready")
    ap.add_argument("--out", default="runs_cls")
    ap.add_argument("--classes", nargs="+", default=["рядовая", "труднообогатимая"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default="runs_cls/best.pt")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inc = tuple(a.classes)
        idxs = sorted(ds.CLASS_TO_IDX[c] for c in inc)
        df = ds.make_splits(ds.load_metadata(a.root), seed=a.seed)
        _, _, te, _ = ds.build_cls_loaders(df, a.root, img_size=a.img_size,
                                           batch_size=a.batch_size, include_classes=inc,
                                           num_workers=a.workers, seed=a.seed)
        model, device, state = load_classifier(a.ckpt, device)
        ys, ps = evaluate(model, te, device)
        _report(ys, ps, idxs, title="TEST (eval-only)")
    else:
        train(a)
