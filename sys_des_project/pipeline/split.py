"""Стратифицированное разбиение train на train/val."""

import json
from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split
import typer

DEFAULT_ANNOTATIONS = Path("data/processed/cifar10/v1/annotations")


def main(
    annotations_dir: Path = typer.Option(DEFAULT_ANNOTATIONS),
    val_size: int = typer.Option(5000, help="Размер валидации"),
    seed: int = typer.Option(42, help="Seed для воспроизводимости"),
) -> None:
    train_json = annotations_dir / "train.json"
    payload = json.loads(train_json.read_text())
    images = payload["images"]
    labels = [img["label"] for img in images]

    train_imgs, val_imgs = train_test_split(
        images,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    # Перезаписываем train.json и создаём val.json
    (annotations_dir / "train.json").write_text(
        json.dumps({"classes": payload["classes"], "images": train_imgs}, indent=2)
    )
    (annotations_dir / "val.json").write_text(
        json.dumps({"classes": payload["classes"], "images": val_imgs}, indent=2)
    )

    logger.info(f"train: {len(train_imgs)}, val: {len(val_imgs)}")


if __name__ == "__main__":
    typer.run(main)
