from __future__ import annotations
import os
import argparse
import json
import torch
import talk_sulfide_confidence as p1
import ore_classifier as clf
from pipeline import (
    sulfide_intergrowth,
    decide_class,
    save_overlay,
    write_geojson,
    CLASS_ORDER,
)
from domain_rules import classify_ore


class OreClassifier:
    """Загружает обе модели один раз и классифицирует любой снимок."""

    def __init__(
        self,
        talc_ckpt: str = "runs_talc/best.pt",
        cls_ckpt: str = "runs_cls/best.pt",
        talc_ref: str = "dataset_ready",
        talc_class_thr: float = 10.0,
        talc_normalize: str = "reinhard",
        talc_downscale: float = 2.0,
        cls_tile: int = 0,
        cls_maxside: int = 12000,
        cls_min_gray: float = 12.0,
        tile: int = 512,
        overlap: int = 128,
        talc_prob_thr: float = 0.5,
        batch_tiles: int = 8,
        uncertain_thr: float = 0.5,
        review_conf: float = 0.60,
        review_uncertain: float = 40.0,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.talc_ref = talc_ref
        self.talc_class_thr = talc_class_thr
        self.talc_normalize = talc_normalize
        self.talc_downscale = talc_downscale
        self.cls_maxside = cls_maxside
        self.cls_min_gray = cls_min_gray
        self.tile = tile
        self.overlap = overlap
        self.talc_prob_thr = talc_prob_thr
        self.batch_tiles = batch_tiles
        self.uncertain_thr = uncertain_thr
        self.review_conf = review_conf
        self.review_uncertain = review_uncertain

        self.talc_model, self.device = p1.load_model(talc_ckpt, self.device)
        print(f"[infer] U-Net талька загружен ({self.device})")

        self.cls_model, self.cls_state = None, None
        if os.path.exists(cls_ckpt):
            self.cls_model, _, self.cls_state = clf.load_classifier(cls_ckpt, self.device)
            cls_list = [self.cls_state["classes"][i] for i in self.cls_state["included_idxs"]]
            print(f"[infer] resnet18 загружен: классы={cls_list} img_size={self.cls_state['img_size']}")
        else:
            print(f"[infer] resnet18 не найден ({cls_ckpt}) — фолбэк на морфоправило")

        self.cls_tile = cls_tile if cls_tile > 0 else (
            clf.detect_crop_tile(talc_ref) if self.cls_model is not None else 0)
        if self.cls_model is not None:
            print(f"[infer] классификация ПО ТАЙЛАМ: тайл={self.cls_tile}px "
                  f"maxside={self.cls_maxside} min_gray={self.cls_min_gray}")

        self.talc_ref_stats = (p1.compute_talc_ref_stats(talc_ref)
                               if talc_normalize == "reinhard" else None)

    def predict(self, image_path: str, save_dir: str | None = None) -> dict:
        """Классифицирует один снимок. Возвращает dict с полями:
        class, class_source, talc_percent, normal_percent, fine_percent,
        sulfide_percent, cls_confidence, votes, mean_confidence,
        uncertain_percent, needs_review, seconds, description.

        Если задан save_dir — пишет маски, overlay и geojson.
        """
        masks_dir = os.path.join(save_dir, "masks") if save_dir else None
        r = p1.process_panorama(
            image_path, self.talc_model, self.device,
            tile=self.tile, overlap=self.overlap,
            talc_prob_thr=self.talc_prob_thr, batch_tiles=self.batch_tiles,
            uncertain_thr=self.uncertain_thr, save_dir=masks_dir,
            tile_progress=False, talc_downscale=self.talc_downscale,
            talc_normalize=self.talc_normalize, talc_ref_stats=self.talc_ref_stats,
        )
        if r is None:
            raise RuntimeError(f"не удалось прочитать снимок: {image_path}")

        morph = sulfide_intergrowth(r["sulfide_mask"])
        cls_name, source, cls_conf, cls_votes = decide_class(
            r["talc_percent"], morph["normal_percent"], morph["fine_percent"],
            image_path, self.cls_model, self.cls_state, self.device,
            self.talc_class_thr, cls_tile=self.cls_tile,
            cls_maxside=self.cls_maxside, cls_min_gray=self.cls_min_gray,
        )
        dr = classify_ore(r["talc_percent"], morph["normal_percent"], morph["fine_percent"])

        if source == "resnet18_tiled":
            _vtxt = ""
            if isinstance(cls_votes, dict):
                _n = cls_votes.get("_n_tiles", 0)
                _vparts = ", ".join(f"{k}:{v}" for k, v in cls_votes.items() if not k.startswith("_"))
                _vtxt = f"; голоса тайлов [{_vparts}] из {_n}"
            desc = (f"Класс {cls_name} (resnet18 по тайлам, ср.уверенность "
                    f"{cls_conf:.2f}{_vtxt}); тальк {r['talc_percent']:.1f}%")
        elif source == "unet_talc":
            desc = f"Оталькованная: тальк {r['talc_percent']:.1f}% > порог {self.talc_class_thr}%"
        else:
            desc = dr["description"]

        needs_review = bool(
            (isinstance(cls_conf, float) and cls_conf < self.review_conf) or
            (r["uncertain_percent"] > self.review_uncertain)
        )

        out = {
            "image": r["image"],
            "image_path": image_path,
            "class": cls_name,
            "class_source": source,
            "talc_percent": r["talc_percent"],
            "normal_percent": round(morph["normal_percent"], 2),
            "fine_percent": round(morph["fine_percent"], 2),
            "sulfide_percent": r["sulfide_percent"],
            "cls_confidence": round(cls_conf, 3) if isinstance(cls_conf, float) else None,
            "votes": cls_votes,
            "mean_confidence": r["mean_confidence"],
            "uncertain_percent": r["uncertain_percent"],
            "needs_review": needs_review,
            "seconds": r["seconds"],
            "description": desc,
        }

        if save_dir:
            stem = os.path.splitext(r["image"])[0]
            geo_dir = os.path.join(save_dir, "geojson")
            ov_dir = os.path.join(save_dir, "overlays")
            os.makedirs(geo_dir, exist_ok=True)
            os.makedirs(ov_dir, exist_ok=True)
            write_geojson(
                os.path.join(geo_dir, f"{stem}.geojson"),
                r["talc_mask"], r["sulfide_mask"],
                {"image": r["image"], "class": cls_name, "talc_percent": r["talc_percent"]},
            )
            overlay_path = os.path.join(ov_dir, f"{stem}_overlay.jpg")
            save_overlay(
                image_path, r["talc_mask"], r["sulfide_mask"], overlay_path,
                f"{cls_name} | talc {r['talc_percent']:.1f}%",
            )
            out["overlay_path"] = overlay_path

        return out


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Инференс одного снимка (AI-классификатор руд)")
    ap.add_argument("--image", required=True, help="путь к снимку")
    ap.add_argument("--talc-ckpt", default="runs_talc/best.pt")
    ap.add_argument("--cls-ckpt", default="runs_cls/best.pt")
    ap.add_argument("--talc-ref", default="dataset_ready")
    ap.add_argument("--talc-normalize", default="reinhard", choices=["none", "bright", "reinhard"])
    ap.add_argument("--talc-downscale", type=float, default=2.0)
    ap.add_argument("--talc-class-thr", type=float, default=10.0)
    ap.add_argument("--cls-tile", type=int, default=0, help="0 = авто")
    ap.add_argument("--cls-maxside", type=int, default=12000)
    ap.add_argument("--cls-min-gray", type=float, default=12.0)
    ap.add_argument("--save-dir", default=None, help="куда писать маски/overlay/geojson (опц.)")
    ap.add_argument("--json", action="store_true", help="печатать результат как JSON")
    return ap


def main():
    args = _build_argparser().parse_args()
    engine = OreClassifier(
        talc_ckpt=args.talc_ckpt, cls_ckpt=args.cls_ckpt, talc_ref=args.talc_ref,
        talc_class_thr=args.talc_class_thr, talc_normalize=args.talc_normalize,
        talc_downscale=args.talc_downscale, cls_tile=args.cls_tile,
        cls_maxside=args.cls_maxside, cls_min_gray=args.cls_min_gray,
    )
    res = engine.predict(args.image, save_dir=args.save_dir)

    if args.json:
        printable = {k: v for k, v in res.items() if k != "votes"}
        printable["votes"] = res.get("votes")
        print(json.dumps(printable, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"снимок:        {res['image']}")
        print(f"КЛАСС:        {res['class']}  (источник: {res['class_source']})")
        print(f"тальк:        {res['talc_percent']:.1f}%")
        print(f"срастания:    ��бычные={res['normal_percent']:.0f}%  тонкие={res['fine_percent']:.0f}%")
        print(f"сульфиды:     {res['sulfide_percent']:.1f}%")
        if res["cls_confidence"] is not None:
            print(f"уверенность:   {res['cls_confidence']:.2f}")
        if isinstance(res["votes"], dict):
            _n = res["votes"].get("_n_tiles", 0)
            _vp = ", ".join(f"{k}:{v}" for k, v in res["votes"].items() if not k.startswith("_"))
            print(f"голоса тайлов: [{_vp}] из {_n}")
        print(f"на проверку:   {'ДА' if res['needs_review'] else 'нет'}")
        print(f"время:        {res['seconds']:.1f}s")
        if res.get("overlay_path"):
            print(f"overlay:      {res['overlay_path']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
