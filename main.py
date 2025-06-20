import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import cv2
import numpy as np
from ultralytics import YOLO


class ModelLoader:
    """Класс загрузки предобученной модели YOLO для детекции объектов.

    Args:
        model_path (Path): Путь к файлу модели (.pt).
        auto_download (bool): Если True, автоматически загружает модель при её отсутствии.

    Raises:
        FileNotFoundError: Если модель не найдена и auto_download=False.
    """
    def __init__(self, model_path: Path, auto_download: bool = True) -> None:
        self.model_path = model_path
        self.model_path.parent.mkdir(exist_ok=True)

        if not auto_download and not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена в {self.model_path}")
        self.model = YOLO(str(self.model_path))


class VideoProcessor:
    """Класс обработки видеофайла: детектирует людей, рисует bounding boxes и маски,
    сохраняет результат.

    Args:
        model: Объект модели YOLO для детекции.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.frame_queue = Queue(maxsize=30)
        self.writer = None
        self.write_future = None

    def safe_release(self) -> None:
        """Безопасное освобождение ресурсов в фоновом потоке"""
        if self.writer:
            self.writer.release()
        self.executor.shutdown(wait=True)

    def process_video(self, input_path: Path, output_path: Path) -> bool:
        """Обрабатывает видеофайл: детекция, аннотация, сохранение.

        Args:
            input_path (Path): Путь к входному видео.
            output_path (Path): Путь для сохранения результата.

        Returns:
            bool: True, если обработка завершена успешно, иначе False.

        Raises:
            FileNotFoundError: Если входной файл не существует (из validate_paths).
            PermissionError: Если нет прав на запись в output_path (из validate_paths).
            RuntimeError: Если не удалось открыть видеофайл или создать VideoWriter.
        """
        self._validate_paths(input_path, output_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видеофайл: {input_path}")
        else:
            print(
                f"Видео успешно открыто. Разрешение: "
                f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)  # кадры в секунду
            frame_size = self._get_frame_size(cap)  # размер кадра
            print(f"Обработка видео: {fps} FPS, размер {frame_size}")

            self.writer = self._create_video_writer(output_path, fps, frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Ошибка создания выходного видеофайла {output_path}")

            self.write_future = self.executor.submit(self.write_frames)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, classes=[0])
                annotated_frame = self.draw_results(frame, results)
                self.frame_queue.put(annotated_frame)

            return True

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return False
        finally:
            self.frame_queue.put(None)
            if hasattr(self, 'write_future') and self.write_future:
                self.write_future.result()
            if 'cap' in locals() and cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    print(f"Ошибка при закрытии VideoCapture: {e}")

            self.executor.submit(self.safe_release)

    def write_frames(self) -> None:
        """Записывает кадры из очереди в выходной видеофайл (работает в фоновом потоке)."""
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            try:
                self.writer.write(frame)
            except Exception as e:
                print(f"Ошибка записи кадра: {e}")

    def draw_results(self, frame, results) -> np.ndarray:
        """Рисует bounding boxes и маски сегментации на кадре.

        Args:
            frame (np.ndarray): Исходный кадр (BGR).
            results: Результаты детекции от YOLO.

        Returns:
            np.ndarray: Кадр с аннотациями.
        """
        for result in results:
            frame = result.plot(img=frame, line_width=1)
            if result.masks is not None:
                for mask in result.masks:
                    contours = self._get_mask_contours(mask.data[0].cpu().numpy(), frame.shape)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        return frame

    def _get_mask_contours(self, mask, frame_shape) -> list:
        """Вычисляет контуры маски сегментации.

        Args:
            mask (np.ndarray): Маска (бинарная матрица).
            frame_shape (tuple): Размеры кадра (H, W, C).

        Returns:
            list: Список контуров для отрисовки.
        """
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _validate_paths(self, input_path: Path, output_path: Path) -> None:
        """Проверяет существование входного файла и права на запись.

        Raises:
            FileNotFoundError: Если входной файл не существует.
            PermissionError: Если нет прав на запись.
        """
        if not input_path.exists():
            raise FileNotFoundError(
                f"Входное видео {input_path.name} не найдено в {input_path.parent}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(output_path.parent, os.W_OK):
            raise PermissionError(f"Нет прав на запись в: {output_path.parent}")

    def _get_frame_size(self, cap: cv2.VideoCapture) -> tuple:
        """Возвращает размер кадра видео в пикселях.

        Args:
            cap: Объект VideoCapture с открытым видеофайлом

        Returns:
            tuple[int, int]: Кортеж вида (ширина, высота) в пикселях
        """
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def _create_video_writer(self, path: Path, fps: float, size: tuple) -> cv2.VideoWriter:
        """Создает VideoWriter с подходящим кодеком.

        Args:
            path (Path): Путь для сохранения.
            fps (float): Частота кадров.
            size (tuple): Размер кадра (ширина, высота).

        Raises:
            ValueError: Если fps или размер некорректны.
            RuntimeError: Если не удалось инициализировать VideoWriter.

        Returns:
            cv2.VideoWriter: Объект для записи видео.
        """
        assert isinstance(path, Path), "path должен быть Path"
        assert fps > 0, "FPS должен быть положительным"

        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Некорректный размер кадра: {size}")

        if path.exists():
            try:
                path.unlink()  # Удаление файла перед созданием нового
                print(f"Удален существующий файл: {path}")
            except Exception as e:
                print(f"Ошибка при удалении файла: {e}")

        for codec in ['mp4v', 'avc1', 'X264']:  # Порядок приоритета
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, size)
            if writer.isOpened():
                print(f"Видео создано с кодеком: {codec}")  # Логирование
                return writer

        # Если ни один кодек не сработал
        raise RuntimeError("Не удалось создать VideoWriter. Проверьте:\n"
                           f"- Путь: {path}\n"
                           f"- FPS: {fps}\n"
                           f"- Размер: {size}")


def main():
    """Точка входа: загружает модель, обрабатывает видео, сохраняет результат."""
    base_dir = Path(__file__).parent
    input_video = base_dir / 'input_data' / 'crowd.mp4'
    output_video = base_dir / 'output_data' / 'result.mp4'
    model_path = base_dir / 'models' / 'yolov8n-seg.pt'

    try:
        model = ModelLoader(model_path).model
        processor = VideoProcessor(model)
        if processor.process_video(input_video, output_video):
            print(f"Завершено. Результат сохранен в {output_video}")
        else:
            print("Обработка видеофайла завершена с ошибками")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()