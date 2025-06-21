# Детекция людей в видео с помощью YOLOv8

Программа для обнаружения людей в видеофайле с отрисовкой bounding boxes и масок сегментации.

## Установка

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/AnastasiiaLanch/Detecting_people_on_video.git
   cd Detecting_people_on_video
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск

```bash
python main.py --input input_data/crowd.mp4 --output output_data/result.mp4
```

**Параметры:**
- `--input`: Путь к входному видео (по умолчанию `input_data/crowd.mp4`)
- `--output`: Папка для результатов (по умолчанию `output_data/`)
- `--model`: Путь к модели (по умолчанию `models/yolov8n-seg.pt`)

## Структура проекта

```
project/
├── main.py               # Главный исполняемый скрипт
├── models/              
│   └── yolov8n-seg.pt    # Веса модели
├── input_data/           
│   └── crowd.mp4         # Исходный видеофайл
├── output_data/      
│   └── result.mp4        # Результат (видеофайл)
├── requirements.txt      # Зависимости
├── README.md             # Инструкции по установке и запуску
├── docs/      
    └── ANALYSIS.md       # Анализ качества и улучшения
```

(При загрузке скрипта main.py отдельно от проекта требуется поместить входной видеофайл crowd.mp4 в папку input_data/, которую надо создать в директории со скриптом)


## Выводы и улучшения

См. docs/ANALYSIS.md

## Примечания

Программа автоматически скачает модель при первом запуске, если она отсутствует.