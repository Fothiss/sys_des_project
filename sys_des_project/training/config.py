"""Разбор и валидация конфига одного запуска."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL_KEYS = {"experiment_name", "run_name", "seed", "data", "model", "train"}


class ConfigError(ValueError):
    """Некорректный конфиг запуска."""


@dataclass(frozen=True)
class DataConfig:
    s3_path: str
    local_dir: Path | None = None
    max_train_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    augment: bool = False
    optimizer: str = "adam"


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    run_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: Path) -> RunConfig:
    """Читает YAML-конфиг и возвращает провалидированный RunConfig."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Конфиг не найден: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Некорректный YAML в {path}: {exc}") from exc

    root = _ensure_mapping(raw, "конфиг")
    _reject_unknown(root, TOP_LEVEL_KEYS, "конфиг")
    for key in ("experiment_name", "run_name", "seed", "data", "model", "train"):
        if key not in root:
            raise ConfigError(f"{key}: обязательное поле")

    return RunConfig(
        experiment_name=_as_name(root["experiment_name"], "experiment_name"),
        run_name=_as_name(root["run_name"], "run_name"),
        seed=_as_int({"seed": root["seed"]}, "seed", where="конфиг"),
        data=_parse_data(root["data"]),
        model=_parse_model(root["model"]),
        train=_parse_train(root["train"]),
    )


def to_dict(cfg: RunConfig) -> dict[str, Any]:
    """Сериализует конфиг в словарь (он же пишется в config.yaml и metrics.json)."""
    return {
        "experiment_name": cfg.experiment_name,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "data": {
            "s3_path": cfg.data.s3_path,
            "local_dir": str(cfg.data.local_dir) if cfg.data.local_dir else None,
            "max_train_samples": cfg.data.max_train_samples,
        },
        "model": {"type": cfg.model.type, "params": cfg.model.params},
        "train": {
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "num_workers": cfg.train.num_workers,
            "augment": cfg.train.augment,
            "optimizer": cfg.train.optimizer,
        },
    }


def flatten(cfg: RunConfig) -> dict[str, str]:
    """Плоские пары key -> str для mlflow.log_params."""
    flat: dict[str, str] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), sub)
        else:
            flat[prefix] = _format_value(value)

    walk("", to_dict(cfg))
    return flat


def _parse_data(raw: Any) -> DataConfig:
    section = _ensure_mapping(raw, "data")
    _reject_unknown(section, {"s3_path", "local_dir", "max_train_samples"}, "data")
    if "s3_path" not in section:
        raise ConfigError("data.s3_path: обязательное поле")

    max_train = section.get("max_train_samples")
    if max_train is not None:
        max_train = _as_int(section, "max_train_samples", where="data")
        if max_train <= 0:
            raise ConfigError("data.max_train_samples: должно быть > 0 или null")

    local_dir = section.get("local_dir")
    return DataConfig(
        s3_path=str(section["s3_path"]).strip().strip("/"),
        local_dir=Path(str(local_dir)) if local_dir else None,
        max_train_samples=max_train,
    )


def _parse_model(raw: Any) -> ModelConfig:
    section = _ensure_mapping(raw, "model")
    _reject_unknown(section, {"type", "params"}, "model")
    model_type = section.get("type")
    if not isinstance(model_type, str) or not model_type:
        raise ConfigError("model.type: обязательное поле (строка)")

    params = section.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError("model.params: ожидался словарь")
    return ModelConfig(type=model_type, params=dict(params))


def _parse_train(raw: Any) -> TrainConfig:
    allowed = {"epochs", "batch_size", "lr", "weight_decay", "num_workers", "augment", "optimizer"}
    section = _ensure_mapping(raw, "train")
    _reject_unknown(section, allowed, "train")

    optimizer = str(section.get("optimizer", "adam"))
    if optimizer not in ("adam", "adamw", "sgd"):
        raise ConfigError(
            f"train.optimizer: поддерживаются adam, adamw, sgd (получено {optimizer})"
        )

    epochs = _as_int(section, "epochs", default=10, where="train")
    batch_size = _as_int(section, "batch_size", default=128, where="train")
    if epochs <= 0:
        raise ConfigError("train.epochs: должно быть > 0")
    if batch_size <= 0:
        raise ConfigError("train.batch_size: должно быть > 0")

    lr = _as_float(section, "lr", default=1e-3, where="train")
    weight_decay = _as_float(section, "weight_decay", default=0.0, where="train")

    return TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=_as_int(section, "num_workers", default=0, where="train"),
        augment=_as_bool(section, "augment", default=False, where="train"),
        optimizer=optimizer,
    )


def _ensure_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: ожидался словарь, получено {type(value).__name__}")
    return value


def _reject_unknown(section: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"{where}: неизвестные ключи {unknown}; допустимы {sorted(allowed)}")


def _as_name(value: Any, where: str) -> str:
    name = str(value).strip()
    if not name:
        raise ConfigError(f"{where}: не может быть пустым")
    if "/" in name:
        raise ConfigError(f"{where}: не может содержать '/' (используется в ключе S3)")
    return name


def _as_int(section: dict[str, Any], key: str, default: int | None = None, where: str = "") -> int:
    if key not in section:
        if default is None:
            raise ConfigError(f"{where}.{key}: обязательное поле")
        return default
    try:
        return int(section[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}.{key}: ожидалось целое число") from exc


def _as_float(
    section: dict[str, Any], key: str, default: float | None = None, where: str = ""
) -> float:
    if key not in section:
        if default is None:
            raise ConfigError(f"{where}.{key}: обязательное поле")
        return default
    try:
        return float(section[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}.{key}: ожидалось число") from exc


def _as_bool(section: dict[str, Any], key: str, default: bool = False, where: str = "") -> bool:
    if key not in section:
        return default
    value = section[key]
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{where}.{key}: ожидалось true/false")


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return str(yaml.safe_dump(list(value), default_flow_style=True)).strip()
    return str(value)
