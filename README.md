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
make setup    # поднять окружение с нуля (venv + все зависимости)
make check    # прогнать линтер и тайпчекер без исправлений
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

--------

