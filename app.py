import os
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import cv2
from PIL import Image

from infer import OreClassifier
from pdf_report import generate_pdf_report
from active_learning import reclassify_from_masks, save_correction


try:
    import streamlit.elements.image as _st_image
    if not hasattr(_st_image, "image_to_url"):
        import base64 as _b64
        import io as _io

        def image_to_url(image, width=-1, clamp=False, channels="RGB",
                         output_format="auto", image_id="", allow_emoji=False):
            try:
                img = image
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype("uint8"))
                if not isinstance(img, Image.Image):
                    return image
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                return "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode()
            except Exception:
                return image

        _st_image.image_to_url = image_to_url
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False


# -----------------------------
# Настройки страницы
# -----------------------------
st.set_page_config(page_title="QC-дашборд микроструктур", layout="wide")
st.title("QC-дашборд: классификация и сегментация микроструктур по SEM/OM-изображениям")


# -----------------------------
# Кэш: инициализация движка
# -----------------------------
@st.cache_resource
def load_engine():
    return OreClassifier(
        talc_ckpt="runs_talc/best.pt",
        cls_ckpt="runs_cls/best.pt",
        talc_ref="dataset_ready",
        talc_class_thr=10.0,
        talc_normalize="reinhard",
        talc_downscale=2.0,
        cls_tile=0,
        cls_maxside=12000,
        cls_min_gray=12.0,
    )

engine = load_engine()


def count_candidates(corr_dir: str) -> int:
    """Сколько исправленных примеров накоплено в пуле коррекций."""
    meta = os.path.join(corr_dir, "metadata.csv")
    if not os.path.exists(meta):
        return 0
    try:
        return int(len(pd.read_csv(meta)))
    except Exception:
        return 0


def launch_retrain(corr_dir: str):
    """Запускает дообучение обеих моделей в фоновом процессе (не блокирует UI)."""
    os.makedirs(corr_dir, exist_ok=True)
    log_path = os.path.join(corr_dir, "retrain.log")
    py = sys.executable or "python"
    cmd = (
        f'"{py}" train_talc_unet.py --root "{corr_dir}" --init-from runs_talc/best.pt '
        f'--freeze-encoder --lr 1e-4 --out runs_talc_ft && '
        f'"{py}" ore_classifier.py --root "{corr_dir}" --init-from runs_cls/best.pt '
        f'--freeze-backbone --lr 3e-5 --out runs_cls_ft'
    )
    logf = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT)


# -----------------------------
# Session state
# -----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = []

if "processed" not in st.session_state:
    st.session_state["processed"] = False


# -----------------------------
# Боковая панель
# -----------------------------
st.sidebar.header("Настройки отображения")
show_overlay = st.sidebar.checkbox("Показать overlay масок", value=True)
show_metrics_table = st.sidebar.checkbox("Показать таблицу фазовых долей", value=True)
show_bar_chart = st.sidebar.checkbox("Показать график фазовых долей", value=True)
st.sidebar.markdown("---")
st.sidebar.header("Active learning")
enable_correction = st.sidebar.checkbox("Режим ручной коррекции масок", value=False)
corrections_dir = st.sidebar.text_input("Папка для обучающих примеров",
                                        value="dataset_ready/corrections")

st.sidebar.markdown("---")
_n_cand = count_candidates(corrections_dir)
if st.sidebar.button("Дообучить модель"):
    if _n_cand == 0:
        st.sidebar.warning("Пул коррекций пуст — сначала сохраните исправленные примеры.")
    else:
        st.session_state["retrain_proc"] = launch_retrain(corrections_dir)
        st.session_state["retrain_log"] = os.path.join(corrections_dir, "retrain.log")
        st.sidebar.success("Дообучение запущено в фоне.")

_proc = st.session_state.get("retrain_proc")
if _proc is not None:
    _rc = _proc.poll()
    if _rc is None:
        st.sidebar.info("⏳ Дообучение выполняется…")
    elif _rc == 0:
        st.sidebar.success("✓ Готово. Новые веса: runs_talc_ft/best.pt, runs_cls_ft/best.pt")
    else:
        st.sidebar.error(f"Ошибка дообучения (код {_rc}). Лог: {st.session_state.get('retrain_log')}")

st.sidebar.caption(
    f"Дообучение может занять от нескольких минут до нескольких часов "
    f"(зависит от GPU и объёма данных) и выполняется в фоне, не блокируя инференс. "
    f"Кандидатов для обучения: {_n_cand}."
)


# -----------------------------
# Загрузка файлов
# -----------------------------
uploaded_files = st.file_uploader(
    "Загрузите одно или несколько изображений",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)


# -----------------------------
# Инференс выполняется только один раз
# -----------------------------
if uploaded_files and not st.session_state["processed"]:

    for uploaded_file in uploaded_files:

        # Чтение изображения
        image = Image.open(uploaded_file).convert("RGB")

        # Временная директория
        tmpdir = tempfile.mkdtemp()
        img_path = os.path.join(tmpdir, uploaded_file.name)
        image.save(img_path)

        # Инференс
        result = engine.predict(img_path, save_dir=tmpdir)

        # Финальный класс
        needs_review = result["needs_review"]
        decision = "auto"
        final_class = result["class"]

        # Сохраняем результат (включая маски для ручной коррекции)
        st.session_state["results"].append({
            "image": img_path,
            "overlay_path": result.get("overlay_path", None),
            "class": final_class,
            "model_class": result["class"],
            "needs_review": bool(needs_review),
            "decision": decision,
            "talc_percent": float(result["talc_percent"]),
            "normal_percent": float(result["normal_percent"]),
            "fine_percent": float(result["fine_percent"]),
            "sulfide_percent": float(result["sulfide_percent"]),
            "uncertain_percent": float(result["uncertain_percent"]),
            "cls_confidence": result["cls_confidence"],
            "seconds": result["seconds"],
            "description": result["description"],
            "class_source": result["class_source"],
            "talc_mask": result.get("talc_mask"),
            "sulfide_mask": result.get("sulfide_mask"),
            "corrected": False,
        })

    st.session_state["processed"] = True
    st.success("Все изображения обработаны.")


# -----------------------------
# Блок ручной коррекции маски (active learning)
# -----------------------------
def correction_block(row, idx):
    st.markdown("#### Ручная коррекция маски (active learning)")

    sulf = row.get("sulfide_mask")
    talc = row.get("talc_mask")
    if sulf is None or talc is None:
        st.warning("Маски недоступны для этого снимка.")
        return

    sulf = np.asarray(sulf)
    talc = np.asarray(talc)
    if sulf.ndim == 3:
        sulf = sulf[..., 0]
    if talc.ndim == 3:
        talc = talc[..., 0]
    H, W = sulf.shape[:2]

    def _reclassify():
        rc = reclassify_from_masks(row["talc_mask"], row["sulfide_mask"])
        row["class"] = rc["class"]
        row["class_source"] = rc["class_source"]
        row["description"] = rc["description"]
        row["talc_percent"] = rc["talc_percent"]
        row["normal_percent"] = rc["normal_percent"]
        row["fine_percent"] = rc["fine_percent"]
        row["sulfide_percent"] = rc["sulfide_percent"]
        row["corrected"] = True
        return rc

    st.markdown("---")
    st.markdown("**Загрузить исправленную маску** (если холст не рисует)")
    with st.container():
        st.caption("Скачайте текущую маску, поправьте её в любом редакторе "
                   "(белое = фаза, чёрное = фон) и загрузите обратно.")

        def _mask_png(m):
            _b = (np.asarray(m) > 127).astype(np.uint8) * 255
            _ok, _buf = cv2.imencode(".png", _b)
            return _buf.tobytes()

        d1, d2 = st.columns(2)
        d1.download_button("Скачать маску сульфидов", _mask_png(sulf),
                           file_name=f"sulfide_mask_{idx}.png", mime="image/png",
                           key=f"dl_sulf_{idx}")
        d2.download_button("Скачать маску талька", _mask_png(talc),
                           file_name=f"talc_mask_{idx}.png", mime="image/png",
                           key=f"dl_talc_{idx}")

        up_sulf = st.file_uploader("Исправленная маска сульфидов (PNG)",
                                   type=["png"], key=f"up_sulf_{idx}")
        up_talc = st.file_uploader("Исправленная маска талька (PNG)",
                                   type=["png"], key=f"up_talc_{idx}")

        if st.button("Применить загруженные маски", key=f"apply_up_{idx}"):
            changed = False
            if up_sulf is not None:
                _a = np.array(Image.open(up_sulf).convert("L").resize((W, H)))
                row["sulfide_mask"] = (_a > 127).astype(np.uint8) * 255
                changed = True
            if up_talc is not None:
                _a = np.array(Image.open(up_talc).convert("L").resize((W, H)))
                row["talc_mask"] = (_a > 127).astype(np.uint8) * 255
                changed = True
            if changed:
                rc = _reclassify()
                st.success(f"Пересчитано: {rc['class']} (тальк {rc['talc_percent']}%, "
                           f"обычные {rc['normal_percent']}%, тонкие {rc['fine_percent']}%)")
            else:
                st.warning("Загрузите хотя бы одну маску (PNG).")

    b1, b2 = st.columns(2)
    if b1.button("Сохранить как обучающий пример", key=f"save_{idx}"):
        info = save_correction(
            row["image"], row["talc_mask"], row["sulfide_mask"],
            ore_class=row["class"], out_root=corrections_dir,
        )
        st.success(f"Сохранено в {info['metadata']} (класс: {info['class']}, "
                   f"тальк {info['talc_percent']}%)")
    if b2.button("Сбросить пометку", key=f"reset_{idx}"):
        row["corrected"] = False
        st.info("Пометка коррекции снята.")


# -----------------------------
# Отображение результатов (всегда из session_state)
# -----------------------------
if st.session_state["results"]:
    st.markdown("---")
    st.subheader("Результаты инференса")

    for idx, row in enumerate(st.session_state["results"]):

        title = os.path.basename(row["image"])
        if row.get("needs_review"):
            title += "  ⚠ на проверку"
        if row.get("corrected"):
            title += "  ✎ исправлено"
        st.markdown(f"### {title}")

        col1, col2 = st.columns(2)

        with col1:
            try:
                st.image(Image.open(row["image"]).convert("RGB"), use_column_width=True)
            except Exception as _e:
                st.warning(f"Не удалось показать изображение: {_e}")
            st.caption("Исходное изображение")

        with col2:
            if show_overlay and row["overlay_path"]:
                try:
                    st.image(Image.open(row["overlay_path"]).convert("RGB"), use_column_width=True)
                except Exception as _e:
                    st.warning(f"Не удалось показать overlay: {_e}")
                st.caption("🟩 зелёный — обычные срастания · 🟥 красный — тонкие срастания · 🟦 синий — тальк")

        # Метрики
        st.write(f"**Класс:** {row['class']}")
        st.write(f"**Класс модели:** {row['model_class']}")
        st.write(f"**Источник:** {row['class_source']}")
        st.write(f"**Описание:** {row['description']}")
        st.write(f"**Уверенность:** {row['cls_confidence']}")
        st.write(f"**Неопределённость:** {row['uncertain_percent']}%")
        st.write(f"**Время инференса:** {row['seconds']} сек")

        phases_df = pd.DataFrame({
            "Фаза": ["Тальк", "Обычные", "Тонкие", "Сульфиды"],
            "Доля (%)": [
                row["talc_percent"],
                row["normal_percent"],
                row["fine_percent"],
                row["sulfide_percent"],
            ],
        })

        if show_metrics_table:
            st.table(phases_df)

        if show_bar_chart:
            st.bar_chart(phases_df.set_index("Фаза"))

        # Блок active learning
        if enable_correction:
            with st.expander("🖌 Коррекция маски / active learning",
                             expanded=bool(row.get("needs_review"))):
                correction_block(row, idx)


# -----------------------------
# Сводный отчёт
# -----------------------------
st.markdown("---")
st.subheader("Сводный отчёт")

# колонки-маски не нужны в таблице/экспорте
_DROP = ["talc_mask", "sulfide_mask"]

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"]).drop(columns=_DROP, errors="ignore")
    df_view = df.copy()
    df_view["image"] = df_view["image"].apply(os.path.basename)
    st.dataframe(df_view, use_container_width=True)
else:
    st.info("Нет данных.")


# -----------------------------
# Экспорт CSV и PDF
# -----------------------------
st.markdown("---")
st.subheader("Экспорт отчётов")

if st.session_state["results"]:

    export_tmp = Path(tempfile.mkdtemp())
    reports_dir = export_tmp / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    export_rows = pd.DataFrame(st.session_state["results"]).drop(columns=_DROP, errors="ignore")

    # CSV
    final_csv = reports_dir / "final_report.csv"
    export_rows.to_csv(final_csv, index=False, encoding="utf-8")

    with open(final_csv, "rb") as f:
        st.download_button(
            "Скачать CSV",
            f.read(),
            file_name="final_report.csv",
            mime="text/csv",
        )

    # PDF
    pdf_path = reports_dir / "final_report.pdf"
    generate_pdf_report(export_rows.to_dict("records"), pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Скачать PDF",
            f.read(),
            file_name="final_report.pdf",
            mime="application/pdf",
        )
