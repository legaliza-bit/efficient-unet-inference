# UNet Inference Benchmark

## Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса UNet-модели на датасете Carvana при использовании различных техник оптимизации. Результатом является бенчмарк, который прогоняется на заранее подготовленном кэшированном датасете для точного замера чистого времени инференса (без учета узкого горлышка дискового I/O).

| # | Пайплайн | Описание |
|---|----------|----------|
| 1 | **PyTorch FP32/FP16 baseline** | Бейзлайновый запуск без оптимизаций |
| 2 | **PyTorch torch.compile** | Графовая компиляция (max-autotune) |
| 3 | **torchao FP8/INT8** | Weight-only FP8 квантизация и статическая INT8 квантизация |
| 4 | **TVM (Relay/LLVM)** | Альтернативный компилятор с AutoTVM-тюнингом |
| 5 | **TensorRT** | NVIDIA TensorRT FP16/INT8/FP8 (опционально) |
| 6 | **Спарсификация** | Magnitude pruning (30%, 50%, 70%) и 2:4 semi-structured |

---

## Архитектура модели

- **Модель**: UNet (milesial/Pytorch-UNet), предобученная на Carvana
- **Параметры**: ~14.3M
- **Scale**: 0.5 (изображения уменьшаются в 2 раза)
- **Задача**: бинарная сегментация (фон / автомобиль)
- **Чекпойнт**: загружается автоматически из [GitHub Releases](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0)

---

## Быстрый старт

### Предварительные требования

- [uv](https://docs.astral.sh/uv/) должен быть установлен
- NVIDIA GPU с драйвером, совместимым с CUDA 12.8
- SM 80+ для INT8 квантизации, SM 89+ для FP8 квантизации

### 1. Установка зависимостей

```bash
uv sync
```

Для TensorRT (опционально):
```bash
uv sync --extra trt
```

### 2. Подготовка кэшированного датасета

Для точных замеров throughput мы кэшируем предобработанные тензоры в оперативную память. Для этого нужно скачать данные и подготовить кэш:

```bash
# Для скачивания нужен настроенный Kaggle API (см. раздел ниже)
uv run python -m src.main --prepare-cache --download
```

Данные будут сохранены в `tmp/bench_cache.pt`. Если этот файл уже есть, бенчмарк подхватит его автоматически.

### 3. Запуск бенчмарка

Полный прогон всех PyTorch-оптимизаций и TensorRT, сохранение результатов в JSON:
```bash
uv run python -m src.main --trt
```

Прогон с альтернативным компилятором TVM (требуется отдельная сборка TVM, см. `TVM_SETUP.md`):
```bash
uv run python -m src.main --tvm
```

Свип по размеру батча:
```bash
uv run python -m src.main --batch-sizes 1 4 8 16
```

Профилирование (chrome traces → `tmp/profiles/`):
```bash
uv run python -m src.main --profile
```

---

## Методология оптимизаций

### PyTorch FP16 Baseline
- Модель переводится в `torch.float16` через `.half()`
- Инференс с `torch.no_grad()` и `torch.amp.autocast("cuda")`
- Outlier trimming: p99 отсечение для расчёта mean throughput, что устраняет влияние случайных фризов ОС.

### torch.compile (max-autotune)
- `torch.compile(model, mode="max-autotune-no-cudagraphs")`
- CUDA Graphs отключены для избежания OOM от private memory pool.
- Для максимальной скорости инференса нужно использовать только один размер батча. Стоит использовать `drop_last=True` в DataLoader, чтобы избежать дорогостоящей рекомпиляции графа на неполном последнем батче.

### torchao FP8 / INT8
- **FP8**: `Float8WeightOnlyConfig` — weight-only квантизация.
- **INT8**: `Int8StaticActivationInt8WeightConfig` — статическая квантизация активаций и весов. Перед инференсом модель прогоняется по `calib_dataloader` для сбора статистик активаций.

### TVM
- ONNX → Relay IR → компиляция под CUDA (с cuDNN/cuBLAS при наличии).
- FP16: `relay.transform.ToMixedPrecision(float16)` — loss-чувствительные операции остаются в FP32.
- AutoTVM: XGBTuner с кэшированием результатов. Запускается как subprocess через `.venv-tvm311/bin/python` (см. `TVM_SETUP.md`).

### Спарсификация
- **Magnitude Pruning**: Глобальный неструктурированный прунинг L1 на уровне весов сверток (уровни: 30%, 50%, 70%).
- **2:4 Semi-structured**: Структурированная спарсификация (аппаратное ускорение поддерживается на архитектурах Ampere+ через cuSPARSELt, однако PyTorch-Conv2d не поддерживает нативный 2:4 speedup "из коробки" без специализированных ядер, поэтому показывается только влияние на качество).

---

## Kaggle API Setup

Перед запуском `--download` необходимо настроить аутентификацию.

1. Зарегистрируйтесь на [Kaggle](https://www.kaggle.com/).
2. Присоединитесь к соревнованию и **примите правила** на странице https://www.kaggle.com/competitions/carvana-image-masking-challenge/rules.
3. Перейдите на https://www.kaggle.com/settings → API → **Create New Token**.
4. Поместите токен в нужную папку:
   - **Если токен OAuth (`KGAT_...`)**: `echo -n 'KGAT_...' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
   - **Если legacy ключ**: `mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

Или используйте переменную окружения `KAGGLE_API_TOKEN='KGAT_...'`.

---

## Структура проекта

```
src/
├── main.py              # Точка входа, CLI, оркестрация пайплайнов
├── config.py            # Пути, гиперпараметры, device
├── model.py             # Загрузка модели, torch.compile, torchao, спарсификация
├── data.py              # CarvanaDataset, кэширование, препроцессинг
├── benchmark.py         # Логика замеров latency/throughput (CUDA Events)
├── metrics.py           # mIoU и Dice через confusion matrix
├── utils.py             # Форматирование и утилиты
├── tvm.py               # Интеграция TVM (subprocess launcher)
├── _tvm_benchmark.py    # Standalone TVM-скрипт (Python 3.11 subprocess)
├── trt.py               # TensorRT: ONNX-экспорт, сборка engine, inference
└── finetune/            # Fine-tuning и QAT
```
