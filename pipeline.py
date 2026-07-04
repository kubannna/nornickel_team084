from __future__ import annotations
import os
import csv
import glob
import json
import time
import argparse
from collections import Counter
import numpy as np
import cv2
import torch
import talk_sulfide_confidence as p1
import ore_classifier as clf
import morphology
from domain_rules import classify_ore

IMG_EXTS = p1.IMG_EXTS
CLASS_ORDER = ["рядовая", "труднообогатимая", "оталькованная"]


def sulfide_intergrowth(sulfide_mask: np.ndarray) -> dict:
    """normal_percent / fine_percent из маски сульфидов (morphology команды)."""
    try:
        objs = morphology.analyze_sulfide_objects(sulfide_mask)
        ratio = morphology.compute_intergrowth_ratio(objs)
        ratio["objects"] = len(objs)
        return ratio
    except Exception as e:
        print(f"   [!] морфология: {e}")
        return {"normal_percent": 0.0, "fine_percent": 0.0, "objects": 0}


def decide_class(talc_percent, normal_percent, fine_percent, image_path,
                 cls_model, cls_state, device, talc_class_thr,
                 cls_tile=1988, cls_maxside=12000, cls_min_gray=12.0):
    """Возвращает (class_name, source, cls_confidence|None, vote_info|None).

    Классификатор применяется ПО ТАЙЛАМ масштаба обучения с агрегацией
    (усреднение вероятностей), а не к целой панораме, ужатой в 256.
    """
    if talc_percent > talc_class_thr:
        return "оталькованная", "unet_talc", None, None
    if cls_model is not None:
        name, prob, votes = clf.predict_class_tiled(
            cls_model, image_path, cls_state["img_size"], device,
            cls_state["included_idxs"],
            tile=cls_tile, maxside=cls_maxside, min_gray=cls_min_gray,
        )
        if name is not None:
            return name, "resnet18_tiled", prob, votes
    dr = classify_ore(talc_percent, normal_percent, fine_percent)
    return dr["class"], "morphology", None, None


def _mask_features(mask, label, max_features=400, epsilon=2.0, min_area=40):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_features]
    feats = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True)
        coords = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(coords) < 3:
            continue
        coords.append(coords[0])
        feats.append({"type": "Feature", "properties": {"phase": label},
                      "geometry": {"type": "Polygon", "coordinates": [coords]}})
    return feats


def write_geojson(path, talc_mask, sulfide_mask, meta):
    fc = {"type": "FeatureCollection", "properties": meta,
          "features": _mask_features(talc_mask, "talc") + _mask_features(sulfide_mask, "sulfide")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def write_csv(results, path):
    cols = ["image", "class", "class_source", "talc_percent", "normal_percent",
            "fine_percent", "sulfide_percent", "cls_confidence", "mean_confidence",
            "uncertain_percent", "needs_review", "seconds", "description"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in cols})


def write_talc_csv(results, path):
    """Совместимость с RUN_REPORT.py: image,talc_percent."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "talc_percent"])
        w.writeheader()
        for r in results:
            w.writerow({"image": r["image"], "talc_percent": r["talc_percent"]})


def write_txt(results, path, dist):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ ПО АНАЛИЗУ РУДНЫХ ШЛИФОВ (AI-классификатор)\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Всего снимков: {len(results)}\n")
        for c in CLASS_ORDER:
            f.write(f"  {c}: {dist.get(c, 0)}\n")
        nr = sum(1 for r in results if r.get("needs_review"))
        f.write(f"  → на проверку эксперту (низкая уверенность): {nr}\n\n")
        f.write("-" * 64 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['image']}  ->  {r['class'].upper()}  [{r['class_source']}]\n")
            f.write(f"   тальк={r['talc_percent']:.2f}%  "
                    f"normal={r['normal_percent']:.1f}%  fine={r['fine_percent']:.1f}%  "
                    f"сульфиды={r['sulfide_percent']:.2f}%\n")
            conf = r.get("cls_confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "—"
            f.write(f"   уверенность класса={conf_s}  "
                    f"неуверенных пикс.={r['uncertain_percent']:.1f}%  "
                    f"{'[→ЭКСПЕРТ]' if r.get('needs_review') else ''}\n")
            f.write(f"   {r['description']}\n")
            f.write("-" * 64 + "\n")


def _register_cyrillic_font():
    """Ищет TTF с кириллицей (reportlab без неё не рендерит русский)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("CyrFont", fp))
                return "CyrFont"
            except Exception:
                continue
    return None


def write_pdf(results, path, dist):
    """PDF через reportlab (best-effort; без кириллического шрифта — пропускаем)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception as e:
        print(f"   [i] reportlab недоступен ({e}) — PDF пропущен, есть report.txt")
        return False
    font = _register_cyrillic_font()
    if font is None:
        print("   [i] нет TTF с кириллицей — PDF пропущен, есть report.txt")
        return False

    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Title"], fontName=font, fontSize=16)
    n = ParagraphStyle("n", parent=styles["Normal"], fontName=font, fontSize=9)
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = [Paragraph("Отчёт по анализу рудных шлифов", h), Spacer(1, 0.4 * cm)]
    summary = f"Всего: {len(results)}   |   " + "   ".join(
        f"{c}: {dist.get(c, 0)}" for c in CLASS_ORDER)
    story += [Paragraph(summary, n), Spacer(1, 0.4 * cm)]

    head = ["#", "Снимок", "Класс", "тальк%", "fine%", "conf", "пров."]
    data = [head]
    for i, r in enumerate(results, 1):
        conf = r.get("cls_confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, float) else "—"
        data.append([str(i), r["image"][:26], r["class"], f"{r['talc_percent']:.1f}",
                     f"{r['fine_percent']:.0f}", conf_s, "!" if r.get("needs_review") else ""])
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
    ]))
    story.append(tbl)
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"   [i] PDF не собран: {e} — есть report.txt")
        return False


def save_overlay(bgr_path, talc_mask, sulfide_mask, out_path, title):
    bgr = p1.seg.imread_unicode(bgr_path)
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        talc_mask = cv2.resize(talc_mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        sulfide_mask = cv2.resize(sulfide_mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    import numpy as np
    from skimage import measure
    ov = bgr.copy()
    binary = (sulfide_mask > 127).astype(np.uint8)
    labeled = measure.label(binary, connectivity=2)
    min_area = 15
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        eq_r = np.sqrt(region.area / np.pi)
        rr = region.perimeter / (2 * np.pi * eq_r) if eq_r > 0 else 1.0
        rr = np.clip(rr, 0, 2) / 2
        is_fine = rr > 0.5
        ys, xs = region.coords[:, 0], region.coords[:, 1]
        ov[ys, xs] = (0, 0, 255) if is_fine else (0, 255, 0)
    ov[talc_mask > 0] = (255, 0, 0)
    out = cv2.addWeighted(bgr, 0.6, ov, 0.4, 0)
    cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    _legend = [("obychnye srastaniya", (0, 255, 0)),
               ("tonkie srastaniya", (0, 0, 255)),
               ("talc", (255, 0, 0))]
    _y = 58
    for _txt, _col in _legend:
        cv2.rectangle(out, (12, _y - 12), (30, _y + 3), _col, -1)
        cv2.putText(out, _txt, (36, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(out, _txt, (36, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        _y += 24
    p1._imwrite_unicode(out_path, out)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = [p for p in sorted(glob.glob(os.path.join(args.images, "*"))) if p.endswith(IMG_EXTS)]
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[!] нет изображений в {args.images}")
        return

    os.makedirs(args.out, exist_ok=True)
    masks_dir = os.path.join(args.out, "masks")
    geo_dir = os.path.join(args.out, "geojson")
    ov_dir = os.path.join(args.out, "overlays")
    for d in (masks_dir, geo_dir, ov_dir):
        os.makedirs(d, exist_ok=True)

    talc_model, device = p1.load_model(args.talc_ckpt, device)
    print(f"[pipeline] U-Net талька загружен ({device})")

    cls_model, cls_state = None, None
    if os.path.exists(args.cls_ckpt):
        cls_model, _, cls_state = clf.load_classifier(args.cls_ckpt, device)
        print(f"[pipeline] resnet18 загружен: классы={[cls_state['classes'][i] for i in cls_state['included_idxs']]} "
              f"img_size={cls_state['img_size']}")
    else:
        print(f"[pipeline] resnet18 не найден ({args.cls_ckpt}) — фолбэк на морфоправило")

    _cls_tile = args.cls_tile if args.cls_tile > 0 else (
        clf.detect_crop_tile(args.talc_ref) if cls_model is not None else 0)
    if cls_model is not None:
        print(f"[pipeline] классификация ПО ТАЙЛАМ: тайл={_cls_tile}px maxside={args.cls_maxside} min_gray={args.cls_min_gray}")

    results = []
    t_all = time.time()
    try:
        from tqdm import tqdm as _tqdm
        _files_iter = _tqdm(list(enumerate(files, 1)), total=len(files), desc="Панорамы", unit="img")
    except Exception:
        _files_iter = enumerate(files, 1)
    _talc_ref_stats = (p1.compute_talc_ref_stats(args.talc_ref)
                       if args.talc_normalize == "reinhard" else None)
    for i, path in _files_iter:
        r = p1.process_panorama(
            path, talc_model, device, tile=args.tile, overlap=args.overlap,
            talc_prob_thr=args.talc_prob_thr, batch_tiles=args.batch_tiles,
            uncertain_thr=args.uncertain_thr, save_dir=masks_dir,
            tile_progress=not args.no_progress, talc_downscale=args.talc_downscale,
            talc_normalize=args.talc_normalize, talc_ref_stats=_talc_ref_stats,
        )
        if r is None:
            continue

        morph = sulfide_intergrowth(r["sulfide_mask"])
        cls_name, source, cls_conf, cls_votes = decide_class(
            r["talc_percent"], morph["normal_percent"], morph["fine_percent"],
            path, cls_model, cls_state, device, args.talc_class_thr,
            cls_tile=_cls_tile, cls_maxside=args.cls_maxside, cls_min_gray=args.cls_min_gray,
        )
        dr = classify_ore(r["talc_percent"], morph["normal_percent"], morph["fine_percent"])
        if source == "resnet18_tiled":
            _vtxt = ""
            if isinstance(cls_votes, dict):
                _n = cls_votes.get("_n_tiles", 0)
                _vparts = ", ".join(f"{k}:{v}" for k, v in cls_votes.items() if not k.startswith("_"))
                _vtxt = f"; голоса тайлов [{_vparts}] из {_n}"
            desc = f"Класс {cls_name} (resnet18 по тайлам, ср.уверенность {cls_conf:.2f}{_vtxt}); тальк {r['talc_percent']:.1f}%"
        elif source == "unet_talc":
            desc = f"Оталькованная: тальк {r['talc_percent']:.1f}% > порог {args.talc_class_thr}%"
        else:
            desc = dr["description"]

        needs_review = bool(
            (isinstance(cls_conf, float) and cls_conf < args.review_conf) or
            (r["uncertain_percent"] > args.review_uncertain)
        )

        rec = {
            "image": r["image"], "image_path": path,
            "class": cls_name, "class_source": source,
            "talc_percent": r["talc_percent"], "sulfide_percent": r["sulfide_percent"],
            "normal_percent": round(morph["normal_percent"], 2),
            "fine_percent": round(morph["fine_percent"], 2),
            "cls_confidence": round(cls_conf, 3) if isinstance(cls_conf, float) else "",
            "mean_confidence": r["mean_confidence"],
            "uncertain_percent": r["uncertain_percent"],
            "needs_review": needs_review,
            "seconds": r["seconds"], "description": desc,
        }
        results.append(rec)

        stem = os.path.splitext(r["image"])[0]
        if not args.no_geojson:
            write_geojson(os.path.join(geo_dir, f"{stem}.geojson"),
                          r["talc_mask"], r["sulfide_mask"],
                          {"image": r["image"], "class": cls_name, "talc_percent": r["talc_percent"]})
        if not args.no_overlay:
            save_overlay(path, r["talc_mask"], r["sulfide_mask"],
                         os.path.join(ov_dir, f"{stem}_overlay.jpg"),
                         f"{cls_name} | talc {r['talc_percent']:.1f}%")

        flag = " [→ЭКСПЕРТ]" if needs_review else ""
        _msg = (f"[{i}/{len(files)}] {r['image']}: {cls_name} [{source}] "
                f"тальк={r['talc_percent']:.1f}% fine={morph['fine_percent']:.0f}% "
                f"({r['seconds']:.1f}s){flag}")
        try:
            from tqdm import tqdm as _tqdm
            _tqdm.write(_msg)
        except Exception:
            print(_msg, flush=True)

    dist = Counter(r["class"] for r in results)
    write_csv(results, os.path.join(args.out, "results.csv"))
    write_talc_csv(results, os.path.join(args.out, "talc_results.csv"))
    write_txt(results, os.path.join(args.out, "report.txt"), dist)
    write_pdf(results, os.path.join(args.out, "report.pdf"), dist)

    dt = time.time() - t_all
    n = len(results)
    avg = dt / n if n else 0.0
    slow = [r["image"] for r in results if r["seconds"] > 300]
    print("\n" + "=" * 60)
    print(f"[pipeline] готово: {n} снимков за {dt:.1f}s (среднее {avg:.1f}s/снимок)")
    for c in CLASS_ORDER:
        print(f"    {c}: {dist.get(c, 0)}")
    print(f"[pipeline] вывод -> {args.out}/ (results.csv, report.txt/pdf, masks/, overlays/, geojson/)")
    if slow:
        print(f"[pipeline][!] превысили 5 мин: {slow}")


def parse_args():
    ap = argparse.ArgumentParser(description="Единый пайплайн: панорама -> класс руды + отчёты")
    ap.add_argument("--images", default="Панорамы")
    ap.add_argument("--out", default="results")
    ap.add_argument("--talc-ckpt", default="runs_talc/best.pt")
    ap.add_argument("--cls-ckpt", default="runs_cls/best.pt")
    ap.add_argument("--talc-class-thr", type=float, default=10.0,
                    help="порог тальк% для класса оталькованная (ЭКСПертное правило = 10)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--talc-prob-thr", type=float, default=0.5)
    ap.add_argument("--batch-tiles", type=int, default=8)
    ap.add_argument("--talc-downscale", type=float, default=1.0,
                    help="ужать панораму в N раз перед U-Net талька (масштаб обуч. тайлов)")
    ap.add_argument("--talc-normalize", default="none",
                    choices=["none", "bright", "clahe", "grayworld", "reinhard"],
                    help="нормализация панорамы перед U-Net талька (реком: reinhard)")
    ap.add_argument("--talc-ref", default="dataset_ready",
                    help="папка обучающих данных для reinhard (LAB-статистика)")
    ap.add_argument("--cls-tile", type=int, default=0,
                    help="размер тайла классификатора (0=авто по медиане кропов dataset_ready)")
    ap.add_argument("--cls-maxside", type=int, default=12000,
                    help="ужать панораму перед тайловой классификацией")
    ap.add_argument("--cls-min-gray", type=float, default=12.0,
                    help="пропускать почти-чёрные тайлы (фон) при классификации")
    ap.add_argument("--uncertain-thr", type=float, default=0.5)
    ap.add_argument("--review-conf", type=float, default=0.60,
                    help="resnet18 conf ниже -> на проверку эксперту")
    ap.add_argument("--review-uncertain", type=float, default=15.0,
                    help="%% неуверенных пикселей выше -> на проверку")
    ap.add_argument("--no-geojson", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-progress", action="store_true", help="отключить прогресс-бар по тайлам")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
