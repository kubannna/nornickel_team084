import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Регистрируем Unicode‑шрифт для кириллицы
pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

PAGE_WIDTH, PAGE_HEIGHT = A4


def scale_image(path, max_width, max_height):
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    ratio = min(max_width / w, max_height / h)
    return int(w * ratio), int(h * ratio)


def generate_pdf_report(results, output_path):
    """
    results: список словарей из session_state["results"]
    output_path: путь к PDF
    """

    c = canvas.Canvas(str(output_path), pagesize=A4)

    # Заголовок отчёта
    c.setFont("DejaVuSans", 20)
    c.drawString(50, PAGE_HEIGHT - 50, "Отчёт классификации руд")

    y = PAGE_HEIGHT - 100

    for row in results:
        # -----------------------------
        # Текстовый блок по изображению
        # -----------------------------
        c.setFont("DejaVuSans", 14)
        c.drawString(50, y, f"Изображение: {os.path.basename(row['image'])}")
        y -= 22

        c.setFont("DejaVuSans", 12)
        c.drawString(50, y, f"Финальный класс: {row['class']}")
        y -= 18

        c.drawString(50, y, f"Класс модели: {row['model_class']}")
        y -= 18

        c.drawString(50, y, f"Нужен ревью: {row['needs_review']}")
        y -= 18

        c.drawString(50, y, f"Решение: {row['decision']}")
        y -= 22

        # Метрики фаз
        c.drawString(
            50,
            y,
            f"Тальк: {row['talc_percent']}%   "
            f"Обычные: {row['normal_percent']}%   "
            f"Тонкие: {row['fine_percent']}%   "
            f"Сульфиды: {row['sulfide_percent']}%",
        )
        y -= 30

        img_path = row["image"]
        overlay_path = row.get("overlay_path", None)

        # -----------------------------
        # Исходное изображение
        # -----------------------------
        if img_path and os.path.exists(img_path):
            w, h = scale_image(img_path, max_width=450, max_height=350)
            c.drawImage(ImageReader(img_path), 50, y - h, width=w, height=h)
            y -= h + 25
        else:
            c.drawString(50, y, "Исходное изображение не найдено.")
            y -= 20

        # -----------------------------
        # Overlay
        # -----------------------------
        if overlay_path and os.path.exists(overlay_path):
            w, h = scale_image(overlay_path, max_width=450, max_height=350)
            c.drawImage(ImageReader(overlay_path), 50, y - h, width=w, height=h)
            y -= h + 35
        else:
            c.drawString(50, y, "Overlay отсутствует.")
            y -= 20

        # -----------------------------
        # Переход на новую страницу
        # -----------------------------
        if y < 150:
            c.showPage()
            c.setFont("DejaVuSans", 12)
            y = PAGE_HEIGHT - 80

    c.save()
