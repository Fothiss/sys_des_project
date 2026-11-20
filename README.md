# sys_des_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study project

## Установка и запуск

### Требования

- Python 3.10
- [uv](https://docs.astral.sh/uv/) — установка:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Быстрый старт

```bash
cp .env.example .env    # затем поменяйте пароли в .env
make setup              # поднять окружение с нуля (venv + все зависимости)
make check              # прогнать линтер и тайпчекер без исправлений
```

`make check` возвращает ненулевой код, если есть ошибки стиля или типов.

### Доступные команды

| Команда | Описание |
|---|---|
| `make setup` | Создать `.venv` и установить runtime + dev зависимости через `uv sync --all-groups` |
| `make check` | Ruff (format + lint) и mypy без исправлений. Успех = код 0, ошибки = ненулевой код |
| `make lint` | Только Ruff, без mypy |
| `make format` | Автоисправление стиля (меняет файлы) |
| `make test` | Запустить pytest |
| `make clean` | Удалить `__pycache__` и `*.pyc` |

### Pre-commit

Хуки устанавливаются один раз:

```bash
uv run pre-commit install
```

После этого при каждом `git commit` автоматически прогоняются `ruff` и `mypy`. Коммит с ошибками стиля или типов блокируется.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sys_des_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sys_des_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sys_des_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Эксперименты

Один запуск = один YAML-конфиг в `configs/`, код при этом не меняется.

```bash
make infra-up                                # S3 + PostgreSQL + MLflow + образ для обучения
make train CONFIG=configs/simple_cnn.yaml    # один запуск в контейнере
make train-local CONFIG=configs/simple_cnn.yaml  # тот же запуск на хосте, для отладки
```

Что делает запуск: скачивает версию данных из S3 (в `data/<s3_path>`, если её там ещё нет),
обучает модель, после каждой эпохи считает на val loss/accuracy/macro-F1, сохраняет лучший
по val loss чекпоинт, в конце замеряет его на test и выкладывает `model.pt`, `config.yaml` и
`metrics.json` в `s3://$S3_BUCKET/models/<experiment_name>/<run_name>/`.

### Результаты экспериментов

Baseline — один линейный слой поверх пикселей (3×32×32 → 10). Сетка — 6 комбинаций
`lr ∈ {0.0003, 0.001, 0.003}` × `batch_size ∈ {64, 128}` на подвыборке 12 000 изображений
и 3 эпохах; все раны сетки лежат в одном эксперименте MLflow `cifar10-grid`.
Конфигурация выбирается **только по val loss**, тест в выборе не участвует.

| Конфигурация | train | эпох | val loss | val acc | val macro-F1 | test acc | test macro-F1 |
|---|---|---|---|---|---|---|---|
| baseline: `linear`, bs128, lr0.001 | 45 000 | 10 | 1.8297 | 0.3758 | 0.3697 | 0.3716 | 0.3668 |
| cnn, **bs64, lr0.001** | 12 000 | 3 | **1.3791** | 0.5080 | 0.4977 | 0.4985 | 0.4866 |
| cnn, bs128, lr0.003 | 12 000 | 3 | 1.4002 | 0.5136 | 0.5028 | 0.4948 | 0.4810 |
| cnn, bs64, lr0.003 | 12 000 | 3 | 1.4722 | 0.4828 | 0.4491 | 0.4665 | 0.4290 |
| cnn, bs128, lr0.001 (полный прогон) | 45 000 | 10 | 1.0009 | 0.6418 | 0.6399 | 0.6441 | 0.6410 |
| **итог: cnn, bs64, lr0.001** | 45 000 | 15 | **0.9771** | **0.6620** | **0.6610** | **0.6554** | **0.6542** |

Свёрточная сеть даёт почти двукратный прирост над линейным baseline: 65.5% против 37.2%
на test. По val loss в сетке выиграла комбинация `bs64, lr0.001` — единственная с val loss
ниже 1.4; шаг `lr=0.0003` проигрывает во всех парах, потому что за 3 эпохи не успевает
сойтись, а `bs64` при равном числе эпох делает вдвое больше шагов обновления и поэтому
сходится быстрее. Итоговый прогон выбранной конфигурации на полных 45 000 изображениях и
15 эпохах дал 66.2% на val и 65.5% на test — расхождение 0.7 п.п., то есть на test модель
переносится без сюрпризов.

Отдельно стоит отметить пункт про выбор чекпоинта: на полном прогоне с bs128/10 эпох
лучшей оказалась 9-я эпоха (val loss 1.0009, 64.2% val accuracy), тогда как после 10-й
эпохи val loss вырос до 1.1629 и accuracy упала до 58.3%. Сохранение лучшего по val loss
чекпоинта вместо последнего дало +6 п.п. на test.

Все раны: MLflow http://localhost:5000 (эксперименты `cifar10-simple-cnn`, `cifar10-baseline`,
`cifar10-grid`), артефакты — `s3://cifar10/models/<experiment_name>/<run_name>/`
(`model.pt`, `config.yaml`, `metrics.json`).

## Данные

Локальное S3-совместимое хранилище — **RustFS**, поднимается через Docker Compose.

```bash
make s3-up      # поднять RustFS + создать бакет cifar10
make s3-down    # остановить (данные в volume сохраняются)
```

**Где посмотреть содержимое:** веб-консоль http://localhost:9001 (логин/пароль из `.env`), S3 API http://localhost:9000, либо через Python:

```bash
uv run python -c "
from sys_des_project.storage import get_s3_client
from sys_des_project.config import S3_BUCKET
s3 = get_s3_client()
for obj in s3.list_objects_v2(Bucket=S3_BUCKET).get('Contents', []):
    print(obj['Key'], obj['Size'])
"
```

**Порядок запуска:**

```bash
cp .env.example .env    # затем поменяйте пароли в .env
make setup      # установить зависимости (uv sync)
make s3-up      # поднять хранилище
make ingest     # скачать CIFAR-10 и залить сырой архив в S3 (идемпотентно)
make data       # конвейер: download -> convert -> split -> stats -> upload
make check      # линтер + тайпчекер
```

`make data` можно запускать с другой версией: `make data VERSION=v2`. Версии в S3 неизменяемы — если `v1` уже существует, `upload` завершится с ошибкой и попросит новую версию.

**Что лежит в S3 после конвейера:**

```text
s3://cifar10/
├── raw/
│   └── cifar-10-python.tar.gz          # исходный архив (~162 MiB)
└── processed/cifar10/v1/
    ├── images/                         # 60 000 PNG (32×32)
    │   ├── train_00000.png ... train_49999.png
    │   └── test_00000.png  ... test_09999.png
    ├── annotations/
    │   ├── train.json                  # 45 000 записей
    │   ├── val.json                    # 5 000 записей
    │   └── test.json                   # 10 000 записей
    ├── stats.json                      # mean/std по каналам (только train)
    └── manifest.json                   # версия, git-коммит, параметры, размеры и sha256 файлов
```

**Карточка датасета:**

| Параметр | Значение |
|---|---|
| Источник | CIFAR-10, https://www.cs.toronto.edu/~kriz/cifar-10.html |
| Лицензия | MIT (см. сайт) |
| Всего изображений | 60 000 |
| Размер изображения | 32×32, RGB |
| Классы | 10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Исходное разбиение | train 50 000 / test 10 000 |
| Наше разбиение | train 45 000 / val 5 000 / test 10 000 (стратифицировано, seed=42) |
| Формат хранения | PNG + JSON-разметка |
| Нормализация | mean ≈ [0.4911, 0.4821, 0.4464], std ≈ [0.2469, 0.2434, 0.2616] (только train) |
| Известные проблемы | Низкое разрешение 32×32; визуально близкие классы (cat/dog, automobile/truck); pickle-формат исходных данных небезопасен — открывать только из проверенного источника (md5 сверяется в ingest/download) |

## Инфраструктура

```bash
make setup        # venv + зависимости (нужны для скрипта сетки)
make infra-up     # S3 (RustFS) + PostgreSQL + MLflow + образ для обучения
make infra-down   # остановить всё
```

Обучение выполняется **в контейнере** `train`, подключённом к той же docker-сети, что S3 и MLflow:

```bash
make train CONFIG=configs/simple_cnn.yaml        # один запуск в контейнере
make experiments CONFIG=configs/grid.yaml        # серия, каждый запуск — отдельный контейнер
make train-local CONFIG=configs/simple_cnn.yaml  # без Docker, для отладки
```

Внутри контейнера `S3_ENDPOINT=http://rustfs:9000` и `MLFLOW_TRACKING_URI=http://mlflow:5000`,
скачанные данные кэшируются в именованном volume `train_data`, git-коммит передаётся через `GIT_COMMIT`.
Образ собирается из `Dockerfile` (база `python:3.10-slim-bookworm`), зависимости ставятся
`uv sync --frozen` из `uv.lock`. После правок кода образ нужно пересобрать: `make infra-up`.

Первый запуск в контейнере скачивает версию данных из S3 в volume `train_data` и занимает
несколько минут; последующие запуски стартуют сразу, а на хосте данные уже лежат в `data/`.

Каждый запуск серии выполняется в **отдельном контейнере** (`docker compose run --rm`):
контейнер создаётся под прогон и удаляется после него, поэтому запуски не делят состояние.

--------

