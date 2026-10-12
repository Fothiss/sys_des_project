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

--------

