import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

from infer import OreClassifier
from pdf_report import generate_pdf_report


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

        # Сохраняем результат
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
        })

    st.session_state["processed"] = True
    st.success("Все изображения обработаны.")


# -----------------------------
# Отображение результатов (всегда из session_state)
# -----------------------------
if st.session_state["results"]:
    st.markdown("---")
    st.subheader("Результаты инференса")

    for row in st.session_state["results"]:

        st.markdown(f"### {os.path.basename(row['image'])}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(row["image"], width=900)

        with col2:
            if show_overlay and row["overlay_path"]:
                st.image(row["overlay_path"], width=900)

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


# -----------------------------
# Сводный отчёт
# -----------------------------
st.markdown("---")
st.subheader("Сводный отчёт")

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"])
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

    # CSV
    final_csv = reports_dir / "final_report.csv"
    pd.DataFrame(st.session_state["results"]).to_csv(final_csv, index=False, encoding="utf-8")

    with open(final_csv, "rb") as f:
        st.download_button(
            "Скачать CSV",
            f.read(),
            file_name="final_report.csv",
            mime="text/csv",
        )

    # PDF
    pdf_path = reports_dir / "final_report.pdf"
    generate_pdf_report(st.session_state["results"], pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Скачать PDF",
            f.read(),
            file_name="final_report.pdf",
            mime="application/pdf",
        )
