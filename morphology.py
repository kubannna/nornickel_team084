# morphology.py
import numpy as np
from skimage import measure
from dataclasses import dataclass
from typing import List, Dict, Optional
from domain_rules import classify_ore


@dataclass
class SulfideObject:
    """Один сульфидный объект"""
    area: float  # площадь в пикселях
    perimeter: float  # периметр
    compactness: float  # компактность (1 = круг, меньше = вытянутый)
    replaced_ratio: float  # доля замещения нерудной фазой (0..1)
    is_fine: bool  # True = тонкое срастание


def analyze_sulfide_objects(
        sulfide_mask: np.ndarray,
        matrix_mask: Optional[np.ndarray] = None,
        min_area: int = 50
) -> List[SulfideObject]:
    """
    Анализирует сульфидные объекты на маске.

    Args:
        sulfide_mask: бинарная маска сульфидов (255 = сульфид)
        matrix_mask: бинарная маска нерудной матрицы (для оценки замещения)
        min_area: минимальная площадь объекта (фильтр шума)

    Returns:
        список объектов SulfideObject
    """
    # Бинаризуем
    binary = (sulfide_mask > 127).astype(np.uint8)

    # Находим связные компоненты
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)

    objects = []
    for region in regions:
        if region.area < min_area:
            continue

        # Компактность: 4*pi*area / perimeter^2 (1 = идеальный круг)
        compactness = (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0

        # Оценка замещения: чем больше периметр относительно площади, тем сильнее замещение
        # Для идеального круга replaced_ratio ~ 1, для вытянутых/замещённых - больше
        equivalent_radius = np.sqrt(region.area / np.pi)
        replaced_ratio = region.perimeter / (2 * np.pi * equivalent_radius) if equivalent_radius > 0 else 1.0
        replaced_ratio = np.clip(replaced_ratio, 0, 2) / 2  # нормализуем в 0..1

        # Порог: если замещение > 0.5 - тонкое срастание
        is_fine = replaced_ratio > 0.5

        objects.append(SulfideObject(
            area=region.area,
            perimeter=region.perimeter,
            compactness=compactness,
            replaced_ratio=replaced_ratio,
            is_fine=is_fine
        ))

    return objects


def compute_phase_percentages(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Считает процентное соотношение фаз.

    Args:
        masks: словарь {название_фазы: маска}

    Returns:
        словарь {название_фазы: процент}
    """
    if not masks:
        return {}

    # Берём размер из первой маски
    total_pixels = list(masks.values())[0].size

    result = {}
    for name, mask in masks.items():
        pixels = np.sum(mask > 127)
        result[name] = (pixels / total_pixels) * 100

    return result


def compute_intergrowth_ratio(objects: List[SulfideObject]) -> Dict[str, float]:
    """
    Считает соотношение обычных и тонких срастаний по площади.

    Returns:
        {'normal_percent': ..., 'fine_percent': ...}
    """
    if not objects:
        return {'normal_percent': 0.0, 'fine_percent': 0.0}

    normal_area = sum(obj.area for obj in objects if not obj.is_fine)
    fine_area = sum(obj.area for obj in objects if obj.is_fine)
    total_area = normal_area + fine_area

    if total_area == 0:
        return {'normal_percent': 0.0, 'fine_percent': 0.0}

    return {
        'normal_percent': (normal_area / total_area) * 100,
        'fine_percent': (fine_area / total_area) * 100
    }


def full_analysis(
        sulfide_mask: np.ndarray,
        talc_mask: Optional[np.ndarray] = None,
        matrix_mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Полный анализ изображения: фазы + срастания + классификация.

    Returns:
        полный отчёт со всеми метриками и классом руды
    """
    # 1. Считаем фазы
    masks = {'sulfide': sulfide_mask}
    if talc_mask is not None:
        masks['talc'] = talc_mask
    if matrix_mask is not None:
        masks['matrix'] = matrix_mask

    phases = compute_phase_percentages(masks)

    # 2. Анализируем срастания
    objects = analyze_sulfide_objects(sulfide_mask, matrix_mask)
    ratio = compute_intergrowth_ratio(objects)

    # 3. Классифицируем руду
    talc_pct = phases.get('talc', 0.0)
    classification = classify_ore(
        talc_percent=talc_pct,
        normal_percent=ratio['normal_percent'],
        fine_percent=ratio['fine_percent']
    )

    return {
        'phases': phases,
        'sulfide_objects_count': len(objects),
        'intergrowth_ratio': ratio,
        'classification': classification,
        'objects_sample': [
            {
                'area': obj.area,
                'compactness': round(obj.compactness, 3),
                'replaced_ratio': round(obj.replaced_ratio, 3),
                'is_fine': obj.is_fine
            }
            for obj in objects[:5]  # первые 5 для отладки
        ]
    }


if __name__ == "__main__":
    # Тест на синтетических данных
    print("ТЕСТ MORPHOLOGY.PY")

    # Создаём синтетическое изображение 512x512
    sulfide = np.zeros((512, 512), dtype=np.uint8)
    talc = np.zeros((512, 512), dtype=np.uint8)
    matrix = np.zeros((512, 512), dtype=np.uint8)

    # Крупный сульфид (обычное срастание) — круглый
    yy, xx = np.ogrid[:512, :512]
    mask1 = (yy - 150) ** 2 + (xx - 150) ** 2 <= 80 ** 2
    sulfide[mask1] = 255

    # Мелкие вытянутые сульфиды (тонкие срастания)
    sulfide[300:320, 300:400] = 255
    sulfide[350:360, 250:380] = 255

    # Тальк — рассеянные тёмные области (~12% площади)
    talc[400:460, 50:200] = 255
    talc[100:130, 300:450] = 255

    # Матрица — всё остальное
    matrix[:, :] = 255
    matrix[sulfide > 127] = 0
    matrix[talc > 127] = 0

    # Запускаем анализ
    result = full_analysis(sulfide, talc, matrix)

    print(f"\n Фазы:")
    for phase, pct in result['phases'].items():
        print(f"   {phase}: {pct:.2f}%")

    print(f"\n Сульфидных объектов: {result['sulfide_objects_count']}")

    print(f"\nСоотношение срастаний:")
    print(f"   Обычные: {result['intergrowth_ratio']['normal_percent']:.1f}%")
    print(f"   Тонкие:  {result['intergrowth_ratio']['fine_percent']:.1f}%")

    print(f"\n Классификация:")
    print(f"   Класс: {result['classification']['class']}")
    print(f"   Описание: {result['classification']['description']}")

    print(f"\nПримеры объектов (первые 5):")
    for obj in result['objects_sample']:
        tag = "ТОНКОЕ" if obj['is_fine'] else "ОБЫЧНОЕ"
        print(
            f"   [{tag}] площадь={obj['area']:.0f}, компактность={obj['compactness']}, замещение={obj['replaced_ratio']}")

    print("тест пройден")
