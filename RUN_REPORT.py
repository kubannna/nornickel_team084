# run_report.py
import csv
import os
from pathlib import Path
from domain_rules import classify_ore
from report_generator import generate_csv_report, generate_txt_report


def generate_real_report(inference_dir: str, output_dir: str):
    """Генерирует отчёт на основе результатов inference_pipeline"""
    inference_path = Path(inference_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Читаем results.csv из inference_pipeline
    results_csv = inference_path / "results.csv"
    if not results_csv.exists():
        print("Файл results.csv не найден. Сначала запусти inference_pipeline.py")
        return

    results = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "image": row["image"],
                "talc_percent": float(row["talc_percent"]),
                "normal_percent": 0.0,  # подставить после обучения модели P1
                "fine_percent": 0.0  # подставить после обучения модели P1
            })

    print(f"Найдено результатов: {len(results)}")

    # Генерируем отчёты
    generate_csv_report(results, str(output_path / "final_report.csv"))
    generate_txt_report(results, str(output_path / "final_report.txt"))

    print(f"\nОтчёты сохранены в: {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    inference_dir = base_dir / "inference_results"
    output_dir = base_dir / "reports"

    generate_real_report(str(inference_dir), str(output_dir))