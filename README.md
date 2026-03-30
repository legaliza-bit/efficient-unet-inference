# UNet Inference Benchmark

## 1. Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса UNet-модели на ImageNet при использовании различных техник оптимизации:

| # | Пайплайн | Описание |
|---|----------|----------|
| 1 | **PyTorch FP16 baseline** | Бейзлайновый запуск без оптимизаций |

---

## 2. Архитектура модели

TBD

---

## 3. PyTorch FP16 Baseline

### Методология

- Модель переводится в `torch.float16` через `.half()`
- Инференс с `torch.no_grad()` и `torch.amp.autocast("cuda")`
- Warmup: 20 итераций для прогрева CUDA-ядер
- Замер GPU latency: CUDA Events (`torch.cuda.Event(enable_timing=True)`)

### Метрики

- **Latency**: среднее, медиана (p50), p95, p99 по батчам
- **Throughput**: `total_samples / total_time`
- **Accuracy**: Top-1, Top-5 на ImageNet val

---
