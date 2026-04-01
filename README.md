# Segmentation Inference Benchmark

## Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса модели сегментации на VOC 2012 при использовании различных техник оптимизации:

| # | Пайплайн               | Описание                                  |
|---|------------------------|-------------------------------------------|
| 1 | **PyTorch FP16 baseline** | Бейзлайновый запуск с FP16 без оптимизаций |

---

## Quickstart

```bash
docker build -t seg-bench .
docker run --gpus all -it seg-bench
```

или без докера

```bash
uv sync
uv run python -m src.main
```

---

## Конфигурация стенда

| Параметр    | Значение                                  |
|-------------|-------------------------------------------|
| **GPU**     | NVIDIA GeForce RTX 4070 SUPER (12 GB)     |
| **CPU**     | Intel Core i5-13400F (13th Gen)            |
| **RAM**     | 21 GB                                      |
| **OS**      | Linux 6.6                                  |
| **Python**  | 3.13.5                                     |
| **PyTorch** | 2.11.0+cu130                               |
| **CUDA**    | 13.0                                       |

---

## Архитектура модели

**DeepLabV3** с backbone **ResNet-50**, предобученная на COCO + VOC labels (`DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`).

- **Задача**: мультиклассовая семантическая сегментация (21 класс VOC)
- **Кол-во параметров**: 42.0M
- **Вход**: `3 × 256 × 256` (нормализован ImageNet mean/std)
- **Выход**: `21 × 256 × 256` (logits по классам)

---

## Данные

- **Датасет**: Pascal VOC 2012 Segmentation, split `val`
- **Кол-во eval-сэмплов**: 200 (фиксированный subset)
- **Предобработка изображений**: Resize → ToTensor → Normalize (ImageNet stats)
- **Предобработка масок**: Resize (nearest) → int64 tensor

---

## PyTorch FP16 Baseline

### Методология

- Модель переводится в `torch.float16` через `.half()`
- Входные тензоры также приводятся к FP16 на GPU
- Инференс с `torch.no_grad()`
- Warmup: 20 итераций для прогрева CUDA-ядер
- Замер GPU latency: CUDA Events (`torch.cuda.Event(enable_timing=True)`)
- Замеры выполняются для batch sizes: **1, 4, 8, 16, 32, 64**

### Результаты

| Batch Size | Latency mean (ms) | Latency p50 (ms) | Latency p95 (ms) | Throughput (samples/s) | mIoU   |
|:----------:|:------------------:|:-----------------:|:-----------------:|:----------------------:|:------:|
| 1          | 9.28               | 8.79              | 12.51             | 107.7                  | 0.6608 |
| 4          | 13.61              | 12.32             | 20.23             | 293.9                  | 0.5605 |
| 8          | 21.35              | 19.83             | 25.42             | 374.8                  | 0.5572 |
| 16         | 35.18              | 35.81             | 40.18             | 437.3                  | 0.5687 |
| 32         | 64.20              | 71.45             | 73.16             | 445.0                  | 0.5784 |
| 64         | 116.78             | 149.42            | 149.62            | 428.2                  | 0.5907 |

**Пиковый throughput**: **445.0 samples/s** при batch_size=32  
**Лучший mIoU**: **0.6608** при batch_size=1  
**Минимальная latency**: **8.79 ms (p50)** при batch_size=1

---

## Структура проекта

```
src/
├── __init__.py           # Версия пакета
├── config.py             # Конфигурация (device, batch sizes, paths)
├── data.py               # VOC 2012 dataset и preprocessing
├── main.py               # Точка входа: запуск бенчмарка
├── run_benchmark.py      # Логика замера latency/throughput/mIoU
├── utils.py              # BenchmarkResult, LatencyTimer, helpers
└── models/
    ├── __init__.py
    └── baseline.py       # DeepLabV3 ResNet50 wrapper
```

---
