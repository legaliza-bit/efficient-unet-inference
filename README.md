# efficient-unet-inference

Минимальный стартовый benchmark для инференса pretrained `smp.Unet` на `OxfordIIITPet`.

Структура:
- `scripts/models.py` готовит конкретные эксперименты;
- `scripts/benchmark_unet.py` отвечает за датасет, замеры, quality-метрики и сохранение результата;
- один запуск бенчмарка = один `json` = одна строка будущей таблицы.

Сейчас доступны эксперименты:
- `baseline_fp32`
- `baseline_fp16`
- `compile`

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt
```

## Запуск

Готовый `Unet` checkpoint на Oxford Pet:

```bash
python scripts/benchmark_unet.py \
  --exp-name baseline_fp32 \
  --download \
  --batch-size 8 \
  --num-samples 64 \
  --image-size 256 \
  --warmup-steps 5 \
  --device cuda \
  --seed 42
```

Pretrained `Unet` checkpoint в `fp16`:

```bash
python scripts/benchmark_unet.py \
  --exp-name baseline_fp16 \
  --download \
  --batch-size 8 \
  --num-samples 64 \
  --image-size 256 \
  --warmup-steps 5 \
  --device cuda \
  --seed 42
```

`torch.compile` на том же pretrained checkpoint:

```bash
python scripts/benchmark_unet.py \
  --exp-name compile \
  --download \
  --batch-size 8 \
  --num-samples 64 \
  --image-size 256 \
  --warmup-steps 5 \
  --device cuda \
  --seed 42
```

Сохраняемый результат имеет вид:
- `exp_name`
- `perf_metrics`
- `quality_metrics`

Quality-метрики сейчас:
- `pixel_accuracy`
- `iou`
- `dice`
