"""Сборка комбинаций из конфига сетки и запуск обучения для каждой."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from loguru import logger
import typer
import yaml

from sys_des_project.config import PROJ_ROOT
from sys_des_project.training.config import load_config

RUNS_DIR = PROJ_ROOT / "runs"
GENERATED_DIR = RUNS_DIR / "grid_configs"
GRID_KEYS = {"base", "overrides", "grid"}
SHORT_KEYS = {
    "lr": "lr",
    "batch_size": "bs",
    "epochs": "ep",
    "weight_decay": "wd",
    "channels": "c",
    "augment": "aug",
    "dropout": "do",
    "max_train_samples": "n",
}


@dataclass(frozen=True)
class GridSpec:
    """Конфиг сетки: базовый конфиг, общие переопределения и оси перебора."""

    base: Path
    overrides: dict[str, Any]
    grid: dict[str, list[Any]]


def load_grid_spec(path: Path) -> GridSpec:
    """Читает и проверяет YAML-конфиг сетки."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - GRID_KEYS)
    if unknown:
        raise ValueError(f"конфиг сетки: неизвестные ключи {unknown}")

    if "base" not in raw:
        raise ValueError("конфиг сетки: обязательное поле base")

    grid = raw.get("grid") or {}
    if not isinstance(grid, dict) or not grid:
        raise ValueError("конфиг сетки: секция grid пуста")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key}: ожидался непустой список значений")

    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise TypeError("конфиг сетки: overrides должен быть словарём")

    base = Path(str(raw["base"]))
    if not base.is_absolute():
        base = PROJ_ROOT / base
    if not base.is_file():
        raise FileNotFoundError(f"Базовый конфиг не найден: {base}")

    return GridSpec(base=base, overrides=dict(overrides), grid=dict(grid))


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Все комбинации значений, порядок детерминированный."""
    keys = sorted(grid)
    value_lists = [grid[key] for key in keys]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*value_lists)]


def apply_overrides(base: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Копия базового конфига со значениями по путям вида 'train.lr'."""
    result = copy.deepcopy(base)
    for dotted, value in values.items():
        _set_dotted(result, dotted, value)
    return result


def build_run_name(combo: dict[str, Any], prefix: str = "grid") -> str:
    """Имя рана по шаблону, например grid-lr0.001-bs128."""
    parts = [prefix]
    for dotted, value in combo.items():
        parts.append(f"{_short_key(dotted)}{_format_value(value)}")
    return "-".join(parts)


def main(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="YAML-конфиг сетки",
    ),
) -> None:
    """Запускает обучение для каждой комбинации гиперпараметров."""
    spec = load_grid_spec(config)
    base_cfg: dict[str, Any] = yaml.safe_load(spec.base.read_text(encoding="utf-8")) or {}
    combos = expand_grid(spec.grid)
    logger.info(f"Сетка: {len(combos)} комбинаций, база {spec.base}")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, combo in enumerate(combos, start=1):
        merged = apply_overrides(base_cfg, {**spec.overrides, **combo})
        merged["run_name"] = build_run_name(combo)
        config_path = GENERATED_DIR / f"{merged['run_name']}.yaml"
        config_path.write_text(
            yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        cfg = load_config(config_path)  # проверяем конфиг до запуска обучения
        logger.info(f"[{index}/{len(combos)}] {cfg.experiment_name}/{cfg.run_name}")
        subprocess.run(_train_command(config_path), check=True)
        results.append(_read_metrics(cfg.experiment_name, cfg.run_name))

    _print_report(results)


def _train_command(config_path: Path) -> list[str]:
    """Команда одного прогона: локально или в отдельном контейнере."""
    if os.getenv("GRID_RUNNER", "local") == "docker":
        return [
            "docker",
            "compose",
            "run",
            "--rm",
            "-T",
            "train",
            "python",
            "-m",
            "sys_des_project.train",
            "--config",
            f"runs/grid_configs/{config_path.name}",
        ]
    return [sys.executable, "-m", "sys_des_project.train", "--config", str(config_path)]


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise TypeError(f"{dotted}: в базовом конфиге нет секции '{part}'")
        node = child
    node[parts[-1]] = value


def _short_key(dotted: str) -> str:
    name = dotted.split(".")[-1]
    return SHORT_KEYS.get(name, name)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _read_metrics(experiment_name: str, run_name: str) -> dict[str, Any]:
    path = RUNS_DIR / experiment_name / run_name / "metrics.json"
    if not path.is_file():
        logger.warning(f"Нет {path} — результат не попал в сводку")
        return {"run_name": run_name}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    val = data.get("val") or {}
    test = data.get("test") or {}
    return {
        "run_name": run_name,
        "best_epoch": data.get("best_epoch"),
        "val_loss": val.get("loss"),
        "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
    }


def _sort_key(item: dict[str, Any]) -> float:
    value = item.get("val_loss")
    return float(value) if value is not None else float("inf")


def _fmt(value: Any, spec: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:{spec}}"
    return "n/a"


def _print_report(results: list[dict[str, Any]]) -> None:
    """Сводка серии, отсортированная по val loss."""
    print()
    print(
        f"{'run_name':40s} {'val_loss':>9s} {'val_acc':>8s} "
        f"{'val_f1':>7s} {'test_acc':>8s} {'epoch':>5s}"
    )
    print("-" * 80)
    for item in sorted(results, key=_sort_key):
        print(
            f"{item['run_name']:40s} "
            f"{_fmt(item.get('val_loss'), '.4f'):>9s} "
            f"{_fmt(item.get('val_accuracy'), '.4f'):>8s} "
            f"{_fmt(item.get('val_macro_f1'), '.4f'):>7s} "
            f"{_fmt(item.get('test_accuracy'), '.4f'):>8s} "
            f"{_fmt(item.get('best_epoch'), 'd'):>5s}"
        )
    print()


if __name__ == "__main__":
    typer.run(main)
