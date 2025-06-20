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

3. Скачайте модель YOLOv8n-seg (если нет автоскачивания):
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt -P models/
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
│   └── result.mp4        # Результ (видеофайл)
├── requirements.txt      # Зависимости
├── README.md             # Инструкции по установке и запуску
├── docs/      
    └── ANALYSIS.md       # Анализ качества иулучшения
```


## Выводы и улучшения

См. docs/ANALYSIS.md