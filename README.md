# UNet Inference Benchmark

## 1. Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса UNet-модели на ImageNet при использовании различных техник оптимизации:

| # | Пайплайн | Описание |
|---|----------|----------|
| 1 | **PyTorch FP16 baseline** | Бейзлайновый запуск без оптимизаций |

---

## 2. Архитектура модели

Residual Encoder–Decoder U-Net c ResNet18 backbone для задачи мультиклассовой сегментации

---

## 3. PyTorch FP16 Baseline

### Методология

- Модель переводится в `torch.float16` через `.half()`
- Инференс с `torch.no_grad()` и `torch.amp.autocast("cuda")`
- Warmup: 20 итераций для прогрева CUDA-ядер
- Замер GPU latency: CUDA Events (`torch.cuda.Event(enable_timing=True)`)

### Метрики

- **Кол-во параметров модели**: 14.3M
- **Latency**: 24.53 ± 10.91 ms (p50=22.32, p95=44.46)
- **Throughput**: 649.0 samples/s

---
