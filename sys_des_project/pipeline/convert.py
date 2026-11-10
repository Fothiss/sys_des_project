"""Конвертация pickle CIFAR-10 в PNG + JSON разметку."""

import hashlib
import json
from pathlib import Path
import pickle
import tarfile
from typing import Any, cast

from loguru import logger
import numpy as np
from PIL import Image
import typer

DEFAULT_ARCHIVE = Path("data/raw/cifar-10-python.tar.gz")
DEFAULT_OUTPUT = Path("data/processed/cifar10/v1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpickle(file_path: Path) -> dict[Any, Any]:
    with file_path.open("rb") as f:
        return cast(dict[Any, Any], pickle.load(f, encoding="bytes"))


def extract_archive(archive: Path, workdir: Path) -> Path:
    """Распаковывает tar.gz, возвращает путь к cifar-10-batches-py."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(workdir)
    return workdir / "cifar-10-batches-py"


def convert_split(
    batch_files: list[Path],
    split_name: str,  # "train" или "test"
    images_dir: Path,
    annotations_dir: Path,
    label_names: list[str],
) -> list[dict[str, Any]]:
    """Конвертирует батчи в PNG, возвращает список записей для JSON."""
    records = []
    global_idx = 0

    for batch_path in batch_files:
        data = unpickle(batch_path)
        images = data[b"data"]  # (N, 3072)
        labels = data[b"labels"]  # (N,)

        for i in range(images.shape[0]):
            # (3072,) -> (3, 32, 32) -> (32, 32, 3)
            img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            pil_img = Image.fromarray(img.astype(np.uint8))

            image_id = f"{split_name}_{global_idx:05d}"
            png_path = images_dir / f"{image_id}.png"
            pil_img.save(png_path)

            label_idx = int(labels[i])
            records.append(
                {
                    "image_id": image_id,
                    "file_name": f"images/{image_id}.png",
                    "label": label_idx,
                    "label_name": label_names[label_idx],
                    "sha256": sha256_file(png_path),
                }
            )
            global_idx += 1

    return records


def main(
    archive: Path = typer.Option(DEFAULT_ARCHIVE),
    output: Path = typer.Option(DEFAULT_OUTPUT),
) -> None:
    workdir = output.parent / "_extract"
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info("Распаковка архива...")
    cifar_dir = extract_archive(archive, workdir)

    meta = unpickle(cifar_dir / "batches.meta")
    label_names = [n.decode() for n in meta[b"label_names"]]

    images_dir = output / "images"
    annotations_dir = output / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Конвертация train...")
    train_files = [cifar_dir / f"data_batch_{i}" for i in range(1, 6)]
    train_records = convert_split(train_files, "train", images_dir, annotations_dir, label_names)

    logger.info("Конвертация test...")
    test_records = convert_split(
        [cifar_dir / "test_batch"], "test", images_dir, annotations_dir, label_names
    )

    payload_train = {"classes": label_names, "images": train_records}
    payload_test = {"classes": label_names, "images": test_records}

    (annotations_dir / "train.json").write_text(json.dumps(payload_train, indent=2))
    (annotations_dir / "test.json").write_text(json.dumps(payload_test, indent=2))

    logger.info(f"Готово: {len(train_records)} train, {len(test_records)} test")


if __name__ == "__main__":
    typer.run(main)
