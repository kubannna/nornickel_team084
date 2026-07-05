Этот проект содержит полный стек для обучения, инференса, морфологического анализа,
классификации руды, пакетной обработки панорам, генерации отчётов и интерактивного
Streamlit‑дашборда.

Все файлы лежат в одной директории.

# АРХИТЕКТУРА ПРОЕКТА
project/
│                                                                                                                                                                                                                             
├── app.py                     # Streamlit‑дашборд                                                                                                                                                                            
├── pdf_report.py              # Генерация PDF (DejaVuSans)                                                                                                                                                                   
├── DejaVuSans.ttf             # Unicode‑шрифт для PDF                                                                                                                                                                        
│
├── pipeline.py                # Главный оркестратор инференса
├── infer.py                   # OreClassifier: тальк + сульфиды + morphology + domain_rules
├── morphology.py              # Анализ сульфидов: connected components, elongation, доли normal/fine
├── domain_rules.py            # Экспертная классификация руды
├── segmentation_sulfide.py    # Сегментация сульфидов (Multi‑Otsu)
├── talc_processor.py          # Детект талька по цвету
├── tile.py                    # Нарезка панорам
├── talk_sulfide_confidence.py # Обработка панорам, confidence, нормализация LAB
│
├── train_talc_unet.py         # Обучение U‑Net талька
├── ore_classifier.py          # Обучение ResNet‑18 классификатора руды
├── load_dataset.py            # Сборка датасета, сплиты, даталоадеры
│
├── dataloader.py              # Загрузчик изображений и масок
├── augmentations.py           # Аугментации
│
├── inference_pipeline.py      # Пакетная обработка панорам
├── report_generator.py        # Генерация CSV/TXT отчётов
│
├── active_learning.py         # Коррекция масок и пересчёт классов
│
├── dataset_ready/
│   ├── images/                # 1179 JPG
│   ├── masks_talc/            # 1179 PNG (42 размечены)
│   └── metadata.csv           # классы руды
│
├── runs_cls/
│   ├── best.pt                # обученная модель классификатора руды
│   └── metrics.json           # метрики обучения (macro‑F1, loss, LR schedule)
│
└── runs_talc/
    └── best.pt                # обученная модель сегментации талька (U‑Net)


# ВАЖНО перед запуском необходимо скачать папки dataset_ready, runs_cls, runs_talc с диска по ссылке 
https://drive.google.com/drive/folders/1cOoSOGNZ7BKPv-L_wE4E9gY_gTISMF2p?usp=drive_link

--------------------------------------------------------------------------------
1. ДАТАСЕТ
--------------------------------------------------------------------------------

Папка dataset_ready/ содержит:
- images/ — 1179 JPG
- masks_talc/ — 1179 PNG (42 размечены, остальные пустые)
- metadata.csv — классы руды

--------------------------------------------------------------------------------
2. ЗАГРУЗЧИКИ И АУГМЕНТАЦИИ
--------------------------------------------------------------------------------

dataloader.py — основной загрузчик:

    from dataloader import get_dataloaders
    train_loader, val_loader = get_dataloaders("dataset_ready", batch_size=8)

Возвращает:
- image: [B, 3, 512, 512]
- mask:  [B, 512, 512]

augmentations.py — аугментации для обучения.

--------------------------------------------------------------------------------
3. МОДУЛИ ИНФЕРЕНСА
--------------------------------------------------------------------------------

morphology.py — анализ сульфидов:
- connected components
- фильтр по площади (MIN_AREA_PX = 33)
- классификация «тонкие/обычные» по elongation (THIN_ELONG = 0.7)
- классификация по перцентилю площади (THIN_AREA_PCTL = 45)
- вычисление долей normal/fine

domain_rules.py — экспертная классификация руды:

    classify_ore(talc_percent, normal_percent, fine_percent)

segmentation_sulfide.py — сегментация сульфидов без обучения:
- Multi‑Otsu
- удаление шума
- remove_small

talc_processor.py — детект талька по цвету (для панорам).

tile.py — нарезка панорам на тайлы.

talk_sulfide_confidence.py — обработка панорам:
- UNet‑тальк
- confidence по бинарной энтропии
- нормализация LAB
- batch‑режим
- compute_talc_ref_stats

--------------------------------------------------------------------------------
4. OreClassifier (infer.py)
--------------------------------------------------------------------------------

OreClassifier объединяет:

- UNet‑тальк
- сегментацию сульфидов
- morphology (normal/fine)
- domain_rules (класс руды)

predict(image_path) возвращает:

{
  class,
  talc_percent,
  normal_percent,
  fine_percent,
  sulfide_percent,
  overlay_path,
  description,
  cls_confidence,
  uncertain_percent,
  seconds
}

--------------------------------------------------------------------------------
5. pipeline.py — главный оркестратор
--------------------------------------------------------------------------------

pipeline.py — полный автоматический пайплайн.

run():

1. идёт по папке со снимками
2. вызывает инференс масок (talc + sulfide)
3. morphology → normal/fine
4. domain_rules → класс руды
5. сохраняет:
   - CSV
   - TXT
   - GeoJSON
   - PDF
   - overlay (alpha‑blend)

save_overlay():
- сульфиды — зелёный
- тонкие срастания — красный
- тальк — синий
- прозрачность — alpha‑blend

--------------------------------------------------------------------------------
6. ОБУЧЕНИЕ МОДЕЛЕЙ
--------------------------------------------------------------------------------

train_talc_unet.py — обучение U‑Net талька:
- encoder: efficientnet‑b0
- BCE + Dice
- tile 512
- early‑stop по MAE ±3%
- метрика: MAE ≈ 1.46 %, within ±3 % = 83 %

ore_classifier.py — обучение классификатора руды:
- ResNet‑18
- weighted CE
- cosine LR
- transfer learning
- macro‑F1 ≈ 0.917

load_dataset.py — сборка датасета:
- metadata
- make_splits (70/15/15)
- TalcSegmentationDataset
- OreClassificationDataset
- class_weights
- build_cls_loaders / build_talc_loaders

--------------------------------------------------------------------------------
7. ОБУЧЕННЫЕ МОДЕЛИ
--------------------------------------------------------------------------------

В проекте присутствуют две директории с обученными весами моделей:

runs_cls/
│
├── best.pt
│     Обученная модель классификации руды (ResNet‑18).
│     Используется в infer.py → OreClassifier для определения класса руды.
│
└── metrics.json
      Метрики обучения классификатора:
      - macro‑F1
      - accuracy
      - confusion matrix
      - параметры обучения (LR, scheduler, веса классов)
      - история лосса и качества по эпохам

runs_talc/
│
└── best.pt
      Обученная модель сегментации талька (U‑Net, encoder efficientnet‑b0).
      Используется в infer.py → OreClassifier для детекта талька:
      - вычисление talc_percent
      - генерация маски талька
      - нормализация цвета (если включена)
      - оценка confidence

Эти модели автоматически загружаются в OreClassifier:

    self.talc_model = load_talc_model("runs_talc/best.pt")
    self.cls_model  = load_cls_model("runs_cls/best.pt")

--------------------------------------------------------------------------------
8. ACTIVE LEARNING
--------------------------------------------------------------------------------

active_learning.py:
- reclassify_from_masks(talc_mask, sulfide_mask)
- save_correction() → dataset_ready/corrections/
- обновляет metadata для fine‑tune

--------------------------------------------------------------------------------
9. STREAMLIT‑ДАШБОРД (app.py)
--------------------------------------------------------------------------------

Функции:
- загрузка одного или нескольких изображений
- инференс через OreClassifier
- отображение:
  - исходника
  - overlay
  - таблицы фазовых долей
  - графиков
- сводный отчёт
- экспорт CSV и PDF

Инференс выполняется один раз, результаты хранятся в session_state.

--------------------------------------------------------------------------------
10. ГЕНЕРАЦИЯ PDF (pdf_report.py)
--------------------------------------------------------------------------------

Использует:
- DejaVuSans.ttf — Unicode‑шрифт
- масштабирование изображений без искажений
- вставку:
  - исходного изображения
  - overlay
  - всех метрик
  - финального класса
  - класса модели
  - решения
  - долей фаз

PDF создаётся из session_state["results"].

--------------------------------------------------------------------------------
11. ПАКЕТНАЯ ОБРАБОТКА ПАНОРАМ
--------------------------------------------------------------------------------

inference_pipeline.py:
- обрабатывает все изображения из папки «Панорамы»
- сохраняет маски в inference_results/
- создаёт results.csv с процентами талька

report_generator.py:
- generate_csv_report()
- generate_txt_report()

--------------------------------------------------------------------------------
12. УСТАНОВКА ЗАВИСИМОСТЕЙ
--------------------------------------------------------------------------------

    pip install -r requirements.txt

--------------------------------------------------------------------------------
13. БЫСТРЫЙ СТАРТ
--------------------------------------------------------------------------------

Пакетная обработка:

    python pipeline.py

Streamlit‑дашборд:

    streamlit run app.py

