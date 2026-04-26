# UNet Inference Benchmark

## Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса UNet-модели на датасете Carvana при использовании различных техник оптимизации:

| # | Пайплайн | Описание |
|---|----------|----------|
| 1 | **PyTorch FP16 baseline** | Бейзлайновый запуск без оптимизаций |
| 2 | **PyTorch torch.compile** | Графовая компиляция (max-autotune) |
| 3 | **torchao FP8/INT8** | Динамическая и статическая квантизация |
| 4 | **TVM (Relay/LLVM)** | Альтернативный компилятор с AutoTVM-тюнингом |
| 5 | **TensorRT** | NVIDIA TensorRT FP16/INT8 (опционально) |

## Итоги (Сравнительная таблица)

*(Запустите `uv run python -m src.main --tvm --tvm-tune` для получения полных метрик)*

| Пайплайн | Precision | Latency (ms) | Throughput (fps) | mIoU | Dice | Комментарий |
|----------|-----------|--------------|------------------|------|------|-------------|
| PyTorch Baseline | FP16 | ~39.42 | ~203 | 0.989 | 0.994 | Без оптимизаций |
| torch.compile | FP16 | ~18.82 | ~425 | 0.989 | 0.994 | max-autotune-no-cudagraphs |
| torchao | FP8 | ~34.60 | ~231 | 0.989 | 0.994 | dynamic act + weight |
| TVM | FP16 | ~36-40 | ~220 | 0.988 | 0.994 | Без тюнинга |

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

### 2. Скачивание датасета

Для скачивания данных используется Kaggle API. Перед запуском `--download` необходимо настроить аутентификацию (см. раздел [Kaggle API Setup](#kaggle-api-setup) ниже).

```bash
uv run python -m src.main --download
```

Данные будут сохранены в `data/carvana/imgs/` и `data/carvana/masks/`.

### 3. Запуск бенчмарка

Полный прогон всех пайплайнов (кроме TVM и TRT):
```bash
uv run python -m src.main
```

С TVM:
```bash
uv run python -m src.main --tvm
```

С TensorRT:
```bash
uv run python -m src.main --trt
```

Профилирование (chrome traces → `tmp/profiles/`):
```bash
uv run python -m src.main --profile
```

---

## Kaggle API Setup

Перед запуском `--download` необходимо настроить аутентификацию.

### Предварительное требование

1. Зарегистрируйтесь на [Kaggle](https://www.kaggle.com/).
2. Присоединитесь к соревнованию и **примите правила** на странице https://www.kaggle.com/competitions/carvana-image-masking-challenge/rules — без этого скачивание будет запрещено.

### Вариант A: API-токен (рекомендуется для kaggle CLI v2+)

1. Перейдите на https://www.kaggle.com/settings → API → **Create New Token**. Файл `kaggle.json` будет загружен на компьютер.

2. Откройте скачанный `kaggle.json` и проверьте значение поля `key`:

   - **Если `key` начинается с `KGAT_`** — это OAuth access token. Его нужно поместить в `~/.kaggle/access_token`:
     ```bash
     mkdir -p ~/.kaggle
     echo -n 'KGAT_your_token_here' > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```

   - **Если `key` — обычная hex-строка** (без префикса `KGAT_`) — это legacy API key. Используйте стандартный метод:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

### Вариант B: Переменная окружения

```bash
export KAGGLE_API_TOKEN='KGAT_your_token_here'
```

###  Важно

Не помещайте токен с префиксом `KGAT_` в поле `key` файла `kaggle.json` — это приведёт к ошибке **401 Unauthorized**. Legacy-метод аутентификации отправляет ключ как HTTP Basic Auth, а `KGAT_`-токены требуют OAuth.

### Проверка настройки

```bash
kaggle competitions list
```

Если команда выводит список соревнований без ошибок — аутентификация настроена правильно.

### Повторная загрузка

Если скачивание прервалось и остался повреждённый zip-файл, скрипт автоматически обнаружит это и повторит загрузку с флагом `--force`.

Альтернативно, данные можно скачать вручную: https://www.kaggle.com/c/carvana-image-masking-challenge.

---

## Запуск TVM (Alternative Compiler)

> ⚠️ **TVM must be built from source before running benchmarks.**
> Pre-built PyPI wheels are CPU-only and will not work with GPU.
> Follow the step-by-step instructions in [TVM_SETUP.md](TVM_SETUP.md) to build TVM with CUDA support.
> The build requires cmake, CUDA toolkit, cuDNN, and takes ~30 minutes.

В проекте используется Apache TVM для генерации эффективных CUDA-ядер. Ввиду требования TVM к Python 3.11, он запускается как отдельный процесс из-под виртуального окружения `.venv-tvm311`.

### 1. Скачивание датасета

TVM требует тот же датасет Carvana. Если датасет ещё не скачан:

```bash
uv run python -m src.main --download
```

### 2. Настройка окружения TVM

```bash
uv python install 3.11
uv venv .venv-tvm311 --python 3.11
uv pip install --python .venv-tvm311/bin/python -r requirements-tvm.txt
```

Подробная инструкция по сборке самого TVM из исходников с поддержкой cuDNN/cuBLAS (для максимальной скорости) находится в `TVM_SETUP.md`.

### 3. Запуск бенчмарка

Прогон TVM пайплайна (FP16/FP32):
```bash
uv run python -m src.main --tvm
```

С применением профилей AutoTVM (заметно ускоряет инференс за счет тюнинга):
> **Внимание:** При первом запуске с флагом `--tvm-tune` процесс тюнинга (подбор оптимальных конфигураций CUDA-ядер) может занять от 1 до 2 часов в зависимости от GPU. Найденные конфигурации будут закэшированы (в `tmp/tvm_cache/` и `tmp/tvm_tuning_logs/`), и все последующие запуски будут использовать этот кэш, занимая всего несколько секунд на загрузку графа.

```bash
uv run python -m src.main --tvm --tvm-tune
```

Количество триалов AutoTVM (по умолчанию 1000):
```bash
uv run python -m src.main --tvm --tvm-tune --tvm-tune-trials 500
```

---

## Запуск TensorRT (опционально)

```bash
uv sync --extra trt
uv run python -m src.main --trt
```

TensorRT запускает два эксперимента: FP16 и INT8 (с калибровкой).

---

## Архитектура модели

- **Модель**: UNet (milesial/Pytorch-UNet), предобученная на Carvana
- **Параметры**: ~14.3M
- **Scale**: 0.5 (изображения уменьшаются в 2 раза)
- **Задача**: бинарная сегментация (фон / автомобиль)
- **Чекпойнт**: загружается автоматически из [GitHub Releases](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0)
- **Batch size**: 8 (по умолчанию, настраивается в `src/config.py`)

---

## PyTorch FP16 Baseline

### Методология

- Модель переводится в `torch.float16` через `.half()`
- Инференс с `torch.no_grad()` и `torch.amp.autocast("cuda")`
- Warmup: 20 итераций для прогрева CUDA-ядер
- Замер GPU latency: CUDA Events (`torch.cuda.Event(enable_timing=True)`)
- Outlier trimming: p99 отсечение для расчёта mean throughput

### Метрики

- **Кол-во параметров модели**: 14.3M
- **Latency**: 24.53 ± 10.91 ms (p50=22.32, p95=44.46)
- **Throughput**: 649.0 samples/s

---

## torch.compile (max-autotune)

### Методология

- `torch.compile(model, mode="max-autotune-no-cudagraphs")`
- CUDA Graphs отключены для избежания OOM от private memory pool
- Пропускаются первые 2 батча (JIT-компиляция)

---

## torchao FP8 / INT8

### FP8

- `Float8DynamicActivationFloat8WeightConfig` — динамическая квантизация активаций и весов в FP8
- Требует SM 8.9+ (Ada Lovelace / Hopper, например RTX 4090)
- Оборачивается в `torch.compile(mode="max-autotune-no-cudagraphs")`

### INT8

- `Int8StaticActivationInt8WeightConfig` — статическая квантизация активаций и весов в INT8
- Калибровка на нескольких батчах train-выборки
- Требует SM 8.0+ (Ampere)
- Оборачивается в `torch.compile(mode="max-autotune-no-cudagraphs")`

---

## TVM

### Методология

- ONNX → Relay IR → компиляция под CUDA (с cuDNN/cuBLAS при наличии)
- FP16: `relay.transform.ToMixedPrecision(float16)` — loss-чувствительные операции остаются в FP32
- Оптимизации Relay: `SimplifyInference`, `FoldConstant`, `FuseOps`, `CombineParallelConv2D`
- AutoTVM: XGBTuner с кэшированием результатов в `tmp/tvm_tuning_logs/`
- Скомпилированные артефакты кэшируются в `tmp/tvm_cache/` (SHA-256 от конфигурации)
- Запускается как subprocess через `.venv-tvm311/bin/python` (TVM требует Python 3.11)
- Два режима замера: compute-only (`module.run()`) и e2e (`set_input + run + get_output`)

---

## CLI-аргументы

| Аргумент | Описание |
|----------|----------|
| `--download` | Скачать датасет Carvana с Kaggle |
| `--tvm` | Запустить TVM FP16/FP32 эксперименты |
| `--tvm-tune` | Включить AutoTVM-тюнинг (медленно при первом запуске) |
| `--tvm-tune-trials N` | Кол-во триалов AutoTVM (по умолчанию 1000) |
| `--trt` | Запустить TensorRT FP16/INT8 эксперименты |
| `--finetune` | Дообучить модель на Carvana |
| `--finetune-qat` | Quantization-Aware Training |
| `--profile` | Сохранить chrome traces в `tmp/profiles/` |

---

## Структура проекта

```
src/
├── main.py              # Точка входа, CLI, оркестрация пайплайнов
├── config.py            # Пути, гиперпараметры, device
├── model.py             # Загрузка модели, torch.compile, torchao-квантизация
├── data.py              # CarvanaDataset, загрузка и препроцессинг
├── run_benchmark.py     # PyTorch-бенчмарк (CUDA Events, warmup, метрики)
├── metrics.py           # mIoU и Dice через confusion matrix
├── utils.py             # BenchmarkResult, загрузка модели, GPU-утилиты
├── tvm.py               # Интеграция TVM (subprocess launcher)
├── _tvm_benchmark.py    # Standalone TVM-скрипт (Python 3.11 subprocess)
├── trt.py               # TensorRT: ONNX-экспорт, сборка engine, inference
├── _trt_build.py        # TensorRT subprocess builder
└── finetune/
    ├── finetune.py      # Fine-tuning и QAT
    └── losses.py        # Функции потерь
```
