"""Статистики нормализации по train."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
from PIL import Image
import typer

DEFAULT_ANNOTATIONS = Path("data/processed/cifar10/v1/annotations")
DEFAULT_IMAGES = Path("data/processed/cifar10/v1/images")
DEFAULT_OUTPUT = Path("data/processed/cifar10/v1/stats.json")


def main(
    annotations_dir: Path = typer.Option(DEFAULT_ANNOTATIONS),
    images_dir: Path = typer.Option(DEFAULT_IMAGES),
    output: Path = typer.Option(DEFAULT_OUTPUT),
) -> None:
    payload = json.loads((annotations_dir / "train.json").read_text())
    records = payload["images"]

    # Аккумулируем суммы и квадраты по каналам
    count = 0
    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)

    for rec in records:
        img = (
            np.asarray(Image.open(images_dir / f"{rec['image_id']}.png"), dtype=np.float64) / 255.0
        )
        sum_ += img.reshape(-1, 3).sum(axis=0)
        sum_sq += (img.reshape(-1, 3) ** 2).sum(axis=0)
        count += img.shape[0] * img.shape[1]

    mean = sum_ / count
    std = np.sqrt(sum_sq / count - mean**2)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": int(count),
    }
    output.write_text(json.dumps(stats, indent=2))
    logger.info(f"mean={mean}, std={std}")


if __name__ == "__main__":
    typer.run(main)
