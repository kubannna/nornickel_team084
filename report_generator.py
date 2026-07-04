# report_generator.py
import csv
import os
from pathlib import Path
from domain_rules import classify_ore


def generate_csv_report(results: list, output_path: str):
    """Генерирует CSV отчёт по результатам анализа панорам"""
    fieldnames = ["image", "class", "talc_percent", "normal_percent", "fine_percent", "description"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig для корректного отображения в Excel
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            classification = classify_ore(
                r.get("talc_percent", 0),
                r.get("normal_percent", 0),
                r.get("fine_percent", 0)
            )
            writer.writerow({
                "image": os.path.basename(r.get("image", "")),
                "class": classification["class"],
                "talc_percent": round(classification["talc_percent"], 2),
                "normal_percent": round(classification["normal_percent"], 2),
                "fine_percent": round(classification["fine_percent"], 2),
                "description": classification["description"]
            })

    print(f"CSV отчёт сохранён: {output_path}")


def generate_txt_report(results: list, output_path: str):
    """Генерирует простой текстовый отчёт (если fpdf не установлен)"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ ПО АНАЛИЗУ РУДНЫХ ШЛИФОВ\n")
        f.write("=" * 60 + "\n\n")

        for i, r in enumerate(results, 1):
            classification = classify_ore(
                r.get("talc_percent", 0),
                r.get("normal_percent", 0),
                r.get("fine_percent", 0)
            )
            f.write(f"Изображение {i}: {os.path.basename(r.get('image', ''))}\n")
            f.write(f"  Класс: {classification['class']}\n")
            f.write(f"  Тальк: {classification['talc_percent']:.2f}%\n")
            f.write(f"  Обычные срастания: {classification['normal_percent']:.2f}%\n")
            f.write(f"  Тонкие срастания: {classification['fine_percent']:.2f}%\n")
            f.write(f"  {classification['description']}\n")
            f.write("-" * 60 + "\n")

    print(f"Текстовый отчёт сохранён: {output_path}")


if __name__ == "__main__":
    # Тест
    test_results = [
        {"image": "panorama1.jpg", "talc_percent": 14, "normal_percent": 20, "fine_percent": 62},
        {"image": "panorama2.jpg", "talc_percent": 5, "normal_percent": 70, "fine_percent": 20},
        {"image": "panorama3.jpg", "talc_percent": 3, "normal_percent": 25, "fine_percent": 65},
    ]
    generate_csv_report(test_results, "test_report.csv")
    generate_txt_report(test_results, "test_report.txt")