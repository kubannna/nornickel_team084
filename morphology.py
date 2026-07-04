import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from domain_rules import classify_ore

MIN_AREA_PX = 33
THIN_ELONG = 0.7
THIN_AREA_PCTL = 45


@dataclass
class SulfideObject:
    """Один сульфидный объект."""
    area: float
    perimeter: float
    solidity: float
    elongation: float
    is_fine: bool


def _region_table(labels):
    """regionprops_table с учётом переименования осей в skimage>=0.26."""
    from skimage import measure
    base = ["area", "perimeter", "solidity"]
    for maj, minr in (("axis_major_length", "axis_minor_length"),
                      ("major_axis_length", "minor_axis_length")):
        try:
            tbl = measure.regionprops_table(
                labels, properties=tuple(base + [maj, minr]))
            return tbl, maj, minr
        except (AttributeError, ValueError):
            continue
    return measure.regionprops_table(labels, properties=tuple(base)), None, None


def _extract_arrays(mask):
    """По-объектные массивы (area, elong, solidity, perimeter) из маски."""
    from skimage import measure
    binary = mask > 127
    labels = measure.label(binary, connectivity=2)
    tbl, maj_key, min_key = _region_table(labels)
    area = np.asarray(tbl["area"], dtype=np.float64)
    perim = np.asarray(tbl["perimeter"], dtype=np.float64)
    solidity = np.asarray(tbl["solidity"], dtype=np.float64)
    if maj_key is not None:
        maj = np.asarray(tbl[maj_key], dtype=np.float64)
        minr = np.asarray(tbl[min_key], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            elong = np.where(maj > 0, 1.0 - minr / maj, 0.0)
    else:
        elong = np.zeros_like(area)
    return area, elong, solidity, perim


def analyze_sulfide_objects(
        sulfide_mask: np.ndarray,
        matrix_mask: Optional[np.ndarray] = None,
        min_area: int = MIN_AREA_PX,
) -> List[SulfideObject]:
    """
    Анализирует сульфидные объекты на маске.

    Объект помечается как тонкое срастание, если он сильно вытянут
    (elong > THIN_ELONG) ИЛИ мелкий (area < перцентиль THIN_AREA_PCTL
    по данному снимку).
    """
    area, elong, solidity, perim = _extract_arrays(sulfide_mask)
    keep = area >= min_area
    area, elong, solidity, perim = area[keep], elong[keep], solidity[keep], perim[keep]

    if area.size == 0:
        return []

    area_thr = np.percentile(area, THIN_AREA_PCTL)
    fine_mask = (elong > THIN_ELONG) | (area < area_thr)

    objects = []
    for a, p, s, e, f in zip(area, perim, solidity, elong, fine_mask):
        objects.append(SulfideObject(
            area=float(a),
            perimeter=float(p),
            solidity=float(s),
            elongation=float(e),
            is_fine=bool(f),
        ))
    return objects


def compute_phase_percentages(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Процентное соотношение фаз (доля пикселей каждой маски)."""
    if not masks:
        return {}
    total_pixels = list(masks.values())[0].size
    result = {}
    for name, mask in masks.items():
        pixels = int(np.sum(mask > 127))
        result[name] = (pixels / total_pixels) * 100
    return result


def compute_intergrowth_ratio(objects: List[SulfideObject]) -> Dict[str, float]:
    """
    Соотношение обычных и тонких срастаний по площади.

    Returns: {'normal_percent': ..., 'fine_percent': ...}
    """
    if not objects:
        return {"normal_percent": 0.0, "fine_percent": 0.0}

    normal_area = sum(o.area for o in objects if not o.is_fine)
    fine_area = sum(o.area for o in objects if o.is_fine)
    total_area = normal_area + fine_area
    if total_area == 0:
        return {"normal_percent": 0.0, "fine_percent": 0.0}

    return {
        "normal_percent": (normal_area / total_area) * 100,
        "fine_percent": (fine_area / total_area) * 100,
    }


def morphology_features(
        objects: List[SulfideObject],
        sulfide_mask: np.ndarray,
        um_per_px: Optional[float] = None,
) -> Dict[str, float]:
    """Агрегированные признаки снимка (как в step2_metrics.py)."""
    binary = sulfide_mask > 127
    total_px = binary.size
    sulfide_px = int(binary.sum())
    base = {
        "sulfide_fraction": round(sulfide_px / total_px, 4) if total_px else 0.0,
        "n_objects": len(objects),
    }
    if not objects:
        base.update({
            "mean_obj_area_px": 0.0, "median_obj_area_px": 0.0,
            "median_obj_area_um2": "", "mean_elongation": 0.0,
            "mean_solidity": 0.0, "median_solidity": 0.0,
            "boundary_density": 0.0, "coarseness_index": 0.0, "thin_frac": 0.0,
        })
        return base

    area = np.array([o.area for o in objects], dtype=np.float64)
    perim = np.array([o.perimeter for o in objects], dtype=np.float64)
    solidity = np.array([o.solidity for o in objects], dtype=np.float64)
    elong = np.array([o.elongation for o in objects], dtype=np.float64)
    fine = np.array([o.is_fine for o in objects], dtype=bool)

    px2 = (um_per_px ** 2) if um_per_px else None
    median_area = float(np.median(area))
    med_solidity = float(np.median(solidity))
    median_area_um2 = median_area * px2 if px2 else None
    coarse_area = median_area_um2 if median_area_um2 is not None else median_area

    base.update({
        "mean_obj_area_px": round(float(area.mean()), 1),
        "median_obj_area_px": round(median_area, 1),
        "median_obj_area_um2": (round(median_area_um2, 1)
                               if median_area_um2 is not None else ""),
        "mean_elongation": round(float(elong.mean()), 3),
        "mean_solidity": round(float(solidity.mean()), 3),
        "median_solidity": round(med_solidity, 3),
        "boundary_density": round(float(perim.sum()) / sulfide_px, 4) if sulfide_px else 0.0,
        "coarseness_index": round(coarse_area * med_solidity, 2),
        "thin_frac": round(float(area[fine].sum() / area.sum()), 4),
    })
    return base


def full_analysis(
        sulfide_mask: np.ndarray,
        talc_mask: Optional[np.ndarray] = None,
        matrix_mask: Optional[np.ndarray] = None,
        um_per_px: Optional[float] = None,
) -> Dict:
    """Полный анализ: фазы + срастания + признаки + классификация."""
    masks = {"sulfide": sulfide_mask}
    if talc_mask is not None:
        masks["talc"] = talc_mask
    if matrix_mask is not None:
        masks["matrix"] = matrix_mask

    phases = compute_phase_percentages(masks)
    objects = analyze_sulfide_objects(sulfide_mask, matrix_mask)
    ratio = compute_intergrowth_ratio(objects)
    feats = morphology_features(objects, sulfide_mask, um_per_px)

    classification = classify_ore(
        talc_percent=phases.get("talc", 0.0),
        normal_percent=ratio["normal_percent"],
        fine_percent=ratio["fine_percent"],
    )

    return {
        "phases": phases,
        "sulfide_objects_count": len(objects),
        "intergrowth_ratio": ratio,
        "features": feats,
        "classification": classification,
        "objects_sample": [
            {
                "area": round(o.area, 1),
                "elongation": round(o.elongation, 3),
                "solidity": round(o.solidity, 3),
                "is_fine": o.is_fine,
            }
            for o in objects[:5]
        ],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТ MORPHOLOGY.PY")
    print("=" * 60)

    sulfide = np.zeros((512, 512), dtype=np.uint8)
    talc = np.zeros((512, 512), dtype=np.uint8)
    matrix = np.zeros((512, 512), dtype=np.uint8)

    yy, xx = np.ogrid[:512, :512]
    mask1 = (yy - 150) ** 2 + (xx - 150) ** 2 <= 80 ** 2
    sulfide[mask1] = 255
    sulfide[300:320, 300:400] = 255
    sulfide[350:360, 250:380] = 255

    talc[400:460, 50:200] = 255
    talc[100:130, 300:450] = 255

    matrix[:, :] = 255
    matrix[sulfide > 127] = 0
    matrix[talc > 127] = 0

    result = full_analysis(sulfide, talc, matrix, um_per_px=1.0)

    print("\nФазы:")
    for phase, pct in result["phases"].items():
        print(f"   {phase}: {pct:.2f}%")
    print(f"\nСульфидных объектов: {result['sulfide_objects_count']}")
    print("\nПризнаки:")
    for k, v in result["features"].items():
        print(f"   {k}: {v}")
    print("\nСоотношение срастаний:")
    print(f"   Обычные: {result['intergrowth_ratio']['normal_percent']:.1f}%")
    print(f"   Тонкие:  {result['intergrowth_ratio']['fine_percent']:.1f}%")
    print("\nКлассификация:")
    print(f"   Класс: {result['classification']['class']}")
    print(f"   Описание: {result['classification']['description']}")
    print("\n" + "=" * 60)
    print("ТЕСТ ПРОЙДЕН")
    print("=" * 60)
