# UNet Inference Benchmark

## Постановка задачи

Цель проекта — измерить и сравнить end-to-end latency и throughput инференса UNet-модели на ImageNet при использовании различных техник оптимизации:

| # | Пайплайн | Описание |
|---|----------|----------|
| 1 | **PyTorch FP16 baseline** | Бейзлайновый запуск без оптимизаций |
| 2 | **PyTorch torch.compile** | Графовая компиляция (max-autotune) |
| 3 | **TVM (Relay/LLVM)** | Альтернативный компилятор с AutoTVM-тюнингом |
| 4 | **torchao FP8/INT8** | Динамическая и статическая квантизация |

## Итоги (Сравнительная таблица)

*(Запустите `uv run python -m src.main --tvm --tvm-tune` для получения полных метрик)*

| Пайплайн | Precision | Latency (ms) | Throughput (fps) | mIoU | Dice | Комментарий |
|----------|-----------|--------------|------------------|------|------|-------------|
| PyTorch Baseline | FP16 | ~39.42 | ~203 | 0.989 | 0.994 | Без оптимизаций |
| torch.compile | FP16 | ~18.82 | ~425 | 0.989 | 0.994 | max-autotune-no-cudagraphs |
| torchao | FP8 | ~34.60 | ~231 | 0.989 | 0.994 | dynamic act + weight |
| TVM | FP16 | ~36-40 | ~220 | 0.988 | 0.994 | Без тюнинга |

---

## Kaggle API Setup

Для скачивания данных используется Kaggle API. Перед запуском `--download` необходимо настроить аутентификацию.

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

В проекте используется Apache TVM для генерации эффективных CUDA-ядер. Ввиду требования TVM к Python 3.11, он запускается как отдельный процесс из-под виртуального окружения `.venv-tvm311`.

### 1. Настройка окружения TVM

```bash
uv python install 3.11
uv venv .venv-tvm311 --python 3.11
.venv-tvm311/bin/pip install -r requirements-tvm.txt
```

Подробная инструкция по сборке самого TVM из исходников с поддержкой cuDNN/cuBLAS (для максимальной скорости) находится в `TVM_SETUP.md`.

### 2. Запуск бенчмарка

Прогон TVM пайплайна (FP16/FP32):
```bash
uv run python -m src.main --tvm
```

С применением профилей AutoTVM (заметно ускоряет инференс за счет тюнинга):
> **Внимание:** При первом запуске с флагом `--tvm-tune` процесс тюнинга (подбор оптимальных конфигураций CUDA-ядер) может занять от 1 до 2 часов в зависимости от GPU. Найденные конфигурации будут закэшированы (в `tmp/tvm_cache/` и `tmp/tvm_tuning_logs/`), и все последующие запуски будут использовать этот кэш, занимая всего несколько секунд на загрузку графа.

```bash
uv run python -m src.main --tvm --tvm-tune
```

## Архитектура модели


---

## PyTorch FP16 Baseline

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
