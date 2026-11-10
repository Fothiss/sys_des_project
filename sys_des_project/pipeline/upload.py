"""Выгрузка обработанных данных в S3 с версией."""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from loguru import logger
import typer

from sys_des_project.storage import get_s3_client

DEFAULT_SRC = Path("data/processed/cifar10/v1")
DEFAULT_BUCKET = "cifar10"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(
    src: Path = typer.Option(DEFAULT_SRC),
    bucket: str = typer.Option(DEFAULT_BUCKET),
    version: str = typer.Option("v1", help="Версия, например v1"),
    val_size: int = typer.Option(5000),
    seed: int = typer.Option(42),
) -> None:
    s3 = get_s3_client()
    prefix = f"processed/cifar10/{version}/"

    # Проверка неизменяемости версии
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if resp.get("KeyCount", 0) > 0:
        logger.error(f"{prefix} уже существует в S3. Укажите новую версию.")
        raise typer.Exit(code=1)

    # Сбор manifest
    annotations = sorted((src / "annotations").glob("*.json"))
    stats_path = src / "stats.json"

    manifest: dict[str, Any] = {
        "version": version,
        "git_commit": git_commit(),
        "params": {"val_size": val_size, "seed": seed},
        "splits": {},
        "files": [],
    }

    for ann in annotations:
        data = json.loads(ann.read_text())
        name = ann.stem  # train / val / test
        manifest["splits"][name] = len(data["images"])
        manifest["files"].append(
            {
                "path": f"annotations/{ann.name}",
                "size": ann.stat().st_size,
                "sha256": sha256_file(ann),
            }
        )

    manifest["files"].append(
        {
            "path": "stats.json",
            "size": stats_path.stat().st_size,
            "sha256": sha256_file(stats_path),
        }
    )

    manifest_path = src / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Загрузка
    logger.info(f"Загружаю {src} -> s3://{bucket}/{prefix}")
    for path in src.rglob("*"):
        if path.is_file():
            key = prefix + str(path.relative_to(src))
            s3.upload_file(str(path), bucket, key)

    logger.info("Готово")


if __name__ == "__main__":
    typer.run(main)
