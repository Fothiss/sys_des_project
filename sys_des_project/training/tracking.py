"""MLflow-трекинг запуска и выгрузка артефактов в S3."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from loguru import logger
import mlflow
import yaml

from sys_des_project.config import S3_BUCKET
from sys_des_project.storage import get_s3_client
from sys_des_project.training.config import RunConfig, flatten, to_dict
from sys_des_project.training.metrics import EvalResult

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def git_commit() -> str:
    """Хеш текущего коммита: из GIT_COMMIT или из git, если он доступен."""
    from_env = os.getenv("GIT_COMMIT")
    if from_env:
        return from_env.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@contextmanager
def start_run(cfg: RunConfig, commit: str) -> Iterator[Any]:
    """Открывает MLflow run, логирует параметры конфига и теги запуска."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags(
            {
                "git_commit": commit,
                "data_version": cfg.data.s3_path,
                "seed": str(cfg.seed),
            }
        )
        yield run


def log_epoch(epoch: int, train_loss: float, val_result: EvalResult) -> None:
    """Метрики одной эпохи: train loss и метрики на val."""
    metrics = {"train/loss": train_loss}
    metrics.update({f"val/{name}": value for name, value in val_result.metrics().items()})
    mlflow.log_metrics(metrics, step=epoch)


def log_summary(
    *,
    best_epoch: int,
    best_val_loss: float,
    val_result: EvalResult,
    test_result: EvalResult,
    train_time_sec: float,
) -> None:
    """Итоги запуска: номер лучшей эпохи, время обучения, метрики лучшей модели."""
    summary: dict[str, float] = {
        "best_epoch": float(best_epoch),
        "best_val_loss": best_val_loss,
        "train_time_sec": train_time_sec,
    }
    summary.update({f"val_best/{name}": value for name, value in val_result.metrics().items()})
    summary.update({f"test/{name}": value for name, value in test_result.metrics().items()})
    mlflow.log_metrics(summary)


def log_confusion(figure: Any) -> None:
    """Матрица ошибок на val как PNG-артефакт в MLflow."""
    mlflow.log_figure(figure, "confusion_matrix.png")


def save_run_artifacts(cfg: RunConfig, local_dir: Path, metrics: dict[str, Any]) -> str:
    """Дописывает config.yaml и metrics.json рядом с model.pt и грузит всё в S3."""
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "config.yaml").write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (local_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    client = get_s3_client()
    prefix = f"models/{cfg.experiment_name}/{cfg.run_name}/"
    for name in ("model.pt", "config.yaml", "metrics.json"):
        path = local_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Нет файла для выгрузки: {path}")
        client.upload_file(str(path), S3_BUCKET, prefix + name)

    uri = f"s3://{S3_BUCKET}/{prefix}"
    logger.info(f"Артефакты выгружены: {uri}")
    return uri
