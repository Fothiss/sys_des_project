# Образ для обучения: зависимости ставятся из uv.lock, база зафиксирована
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir uv

WORKDIR /app

# Сначала зависимости: слой переиспользуется, пока не изменился lock-файл
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

CMD ["python", "-m", "sys_des_project.train", "--help"]