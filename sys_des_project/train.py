"""Один запуск эксперимента: обучение, метрики, MLflow, выгрузка модели в S3."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import time
from typing import Any

from loguru import logger
import mlflow
import numpy as np
import torch
from torch import nn
import typer

from sys_des_project.config import PROJ_ROOT
from sys_des_project.training import tracking
from sys_des_project.training.config import ConfigError, RunConfig, load_config, to_dict
from sys_des_project.training.data import (
    build_datasets,
    ensure_dataset,
    make_loader,
    resolve_local_dir,
)
from sys_des_project.training.metrics import confusion_figure, confusion_matrix_list, evaluate
from sys_des_project.training.models import build_model

RUNS_DIR = PROJ_ROOT / "runs"


def main(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="YAML-конфиг одного запуска",
    ),
) -> None:
    """Обучает модель по конфигу и логирует запуск в MLflow."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        logger.error(f"Ошибка конфига: {exc}")
        raise typer.Exit(code=1) from exc

    logger.info(f"Запуск {cfg.experiment_name}/{cfg.run_name} (seed={cfg.seed})")
    set_seed(cfg.seed)

    data_root = resolve_local_dir(cfg.data)
    ensure_dataset(cfg.data, data_root)
    datasets = build_datasets(data_root, cfg.data, cfg.train, cfg.seed)
    num_classes = len(datasets.classes)

    train_loader = make_loader(
        datasets.train,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        datasets.val,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        datasets.test,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    model = build_model(cfg.model, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)

    run_dir = RUNS_DIR / cfg.experiment_name / cfg.run_name
    checkpoint_path = run_dir / "model.pt"
    commit = tracking.git_commit()

    with tracking.start_run(cfg, commit) as run:
        best_epoch = 0
        best_val_loss = float("inf")
        history: list[dict[str, float]] = []
        started = time.perf_counter()

        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_result = evaluate(model, val_loader, criterion)
            entry: dict[str, float] = {"epoch": epoch, "train_loss": train_loss}
            entry.update({f"val_{name}": value for name, value in val_result.metrics().items()})
            history.append(entry)
            tracking.log_epoch(epoch, train_loss, val_result)
            logger.info(
                f"эпоха {epoch}/{cfg.train.epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} "
                f"val_f1={val_result.macro_f1:.4f}"
            )
            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                best_epoch = epoch
                save_checkpoint(model, checkpoint_path, cfg, epoch)
                logger.info(f"лучший чекпоинт: эпоха {epoch}, val_loss={best_val_loss:.4f}")

        train_time_sec = time.perf_counter() - started
        if best_epoch == 0:
            raise RuntimeError("Лучший чекпоинт не сохранён: ни одна эпоха не отработала")

        # Test — только итоговый замер лучшей модели, в выборе конфигурации не участвует
        load_checkpoint(model, checkpoint_path)
        best_val = evaluate(model, val_loader, criterion)
        test_result = evaluate(model, test_loader, criterion)
        logger.info(f"test лучшей модели (эпоха {best_epoch}): {test_result.metrics()}")

        figure = confusion_figure(best_val.targets, best_val.preds, datasets.classes)
        tracking.log_confusion(figure)
        tracking.log_summary(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            val_result=best_val,
            test_result=test_result,
            train_time_sec=train_time_sec,
        )

        metrics: dict[str, Any] = {
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "git_commit": commit,
            "data": {
                "s3_path": cfg.data.s3_path,
                "local_dir": str(data_root),
                "max_train_samples": cfg.data.max_train_samples,
                "n_train": len(datasets.train),
                "n_val": len(datasets.val),
                "n_test": len(datasets.test),
            },
            "epochs": cfg.train.epochs,
            "train_time_sec": train_time_sec,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "val": best_val.metrics(),
            "test": test_result.metrics(),
            "val_confusion_matrix": confusion_matrix_list(
                best_val.targets, best_val.preds, num_classes
            ),
            "mlflow_run_id": run.info.run_id,
        }
        uri = tracking.save_run_artifacts(cfg, run_dir, metrics)
        mlflow.set_tag("s3_model_uri", uri)

    logger.info(f"Готово. MLflow run: {run.info.run_id}")


def set_seed(seed: int) -> None:
    """Фиксирует seed там, где он влияет на обучение."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Одна эпоха на train; возвращает средний loss по сэмплам."""
    model.train()
    total_loss = 0.0
    total = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        batch_size = int(inputs.size(0))
        total_loss += float(loss.item()) * batch_size
        total += batch_size
    if total == 0:
        raise RuntimeError("Пустой train: обучать не на чем")
    return total_loss / total


def make_optimizer(cfg: RunConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Оптимизатор из конфига."""
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=0.9,
        weight_decay=cfg.train.weight_decay,
    )


def save_checkpoint(model: nn.Module, path: Path, cfg: RunConfig, epoch: int) -> None:
    """Сохраняет состояние модели лучшей по val loss эпохи."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "config": to_dict(cfg)},
        path,
    )


def load_checkpoint(model: nn.Module, path: Path) -> None:
    """Загружает лучший чекпоинт для итогового замера на test."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])


if __name__ == "__main__":
    typer.run(main)
