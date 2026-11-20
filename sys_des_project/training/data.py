"""Скачивание версии данных из S3 и датасеты для обучения."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from botocore.client import BaseClient
from botocore.exceptions import ClientError
from loguru import logger
import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from sys_des_project.config import PROJ_ROOT, S3_BUCKET
from sys_des_project.storage import get_s3_client
from sys_des_project.training.config import DataConfig, TrainConfig

DATA_ROOT = PROJ_ROOT / "data"
MARKER_NAME = ".download_complete.json"
DOWNLOAD_WORKERS = 8
IMAGE_SHAPE = (32, 32, 3)
META_FILES = ("manifest.json", "stats.json")
ANNOTATION_SPLITS = ("train", "val", "test")


def resolve_local_dir(data_cfg: DataConfig) -> Path:
    """Куда класть версию данных: путь из конфига или data/<s3_path>."""
    if data_cfg.local_dir is not None:
        if data_cfg.local_dir.is_absolute():
            return data_cfg.local_dir
        return PROJ_ROOT / data_cfg.local_dir
    return DATA_ROOT / data_cfg.s3_path


def ensure_dataset(data_cfg: DataConfig, local_root: Path, *, bucket: str = S3_BUCKET) -> Path:
    """Скачивает версию из S3, если её ещё нет локально, и возвращает путь."""
    prefix = data_cfg.s3_path.strip("/") + "/"
    marker = local_root / MARKER_NAME
    if marker.is_file():
        logger.info(f"Версия {data_cfg.s3_path} уже на диске: {local_root}")
        return local_root
    if _local_version_complete(local_root):
        logger.info(f"Версия {data_cfg.s3_path} уже на диске: {local_root} (проверено локально)")
        _write_marker(marker, prefix, "local")
        return local_root

    local_root.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()

    # Метаданные и разметка — по известным ключам, без листинга бакета
    meta_keys = [*META_FILES, *(f"annotations/{split}.json" for split in ANNOTATION_SPLITS)]
    for key in meta_keys:
        target = local_root / key
        if target.is_file():
            continue
        logger.info(f"Скачиваю s3://{bucket}/{prefix}{key}")
        try:
            _download_one(client, bucket, prefix + key, target)
        except ClientError as exc:
            raise FileNotFoundError(
                f"В s3://{bucket}/{prefix} нет файла {key}. "
                "Соберите версию данных командами из README (make ingest, затем make data)."
            ) from exc

    # Список изображений берём из разметки: она ссылается на каждый файл
    expected = _expected_images(local_root)
    missing = [path for path in expected if not (local_root / path).is_file()]
    logger.info(f"Изображений в версии: {len(expected)}, скачать: {len(missing)}")
    if missing:
        _download_many(client, bucket, prefix, local_root, missing)

    _write_marker(marker, prefix, "s3")
    logger.info(f"Данные готовы: {local_root}")
    return local_root


def load_stats(root: Path) -> tuple[list[float], list[float]]:
    """mean и std по каналам из stats.json той же версии данных."""
    stats_path = root / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Нет stats.json: {stats_path}")
    stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = [float(value) for value in stats["mean"]]
    std = [float(value) for value in stats["std"]]
    return mean, std


def select_indices(labels: list[int], max_samples: int | None, seed: int) -> list[int]:
    """Индексы сэмплов; при ограничении — стратифицированный сабсет по seed."""
    indices = list(range(len(labels)))
    if max_samples is None or max_samples >= len(labels):
        return indices
    if max_samples <= 0:
        raise ValueError("max_train_samples должен быть положительным")

    subset, _ = train_test_split(
        indices,
        train_size=max_samples,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return sorted(int(index) for index in subset)


@dataclass(frozen=True)
class Datasets:
    """Три готовых сплита и список классов."""

    train: CifarDataset
    val: CifarDataset
    test: CifarDataset
    classes: list[str]


class CifarDataset(Dataset[tuple[torch.Tensor, int]]):
    """PNG + JSON-разметка; изображения декодируются один раз в память."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        mean: list[float],
        std: list[float],
        augment: bool = False,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        payload = _read_annotations(root, split)
        self.split = split
        self.classes: list[str] = [str(name) for name in payload["classes"]]
        records: list[dict[str, Any]] = list(payload["images"])

        labels = [int(record["label"]) for record in records]
        indices = select_indices(labels, max_samples, seed)
        self.records = [records[index] for index in indices]
        self.labels = [labels[index] for index in indices]
        self.images = _load_images(root, self.records, split)

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.augmentations: list[Any] = []
        if augment:
            self.augmentations = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float().div_(255.0)
        for operation in self.augmentations:
            image = operation(image)
        image = (image - self.mean) / self.std
        return image, self.labels[index]


def build_datasets(
    root: Path,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> Datasets:
    """Три сплита с нормализацией по stats.json."""
    mean, std = load_stats(root)
    train_dataset = CifarDataset(
        root,
        "train",
        mean=mean,
        std=std,
        augment=train_cfg.augment,
        max_samples=data_cfg.max_train_samples,
        seed=seed,
    )
    val_dataset = CifarDataset(root, "val", mean=mean, std=std)
    test_dataset = CifarDataset(root, "test", mean=mean, std=std)
    logger.info(
        f"Сплиты: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return Datasets(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        classes=train_dataset.classes,
    )


def make_loader(
    dataset: CifarDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """DataLoader поверх in-memory датасета."""
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def _local_version_complete(root: Path) -> bool:
    """Локальная копия полная: есть stats.json и все PNG, на которые ссылается разметка."""
    if not (root / "stats.json").is_file():
        return False
    for split in ("train", "val", "test"):
        path = root / "annotations" / f"{split}.json"
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("images", []):
            if not (root / str(record["file_name"])).is_file():
                return False
    return True


def _expected_images(root: Path) -> list[str]:
    """Относительные пути всех изображений, на которые ссылается разметка."""
    expected: list[str] = []
    for split in ANNOTATION_SPLITS:
        payload = _read_annotations(root, split)
        expected.extend(str(record["file_name"]) for record in payload["images"])
    return expected


def _download_one(client: BaseClient, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


def _download_many(
    client: BaseClient,
    bucket: str,
    prefix: str,
    local_root: Path,
    paths: list[str],
) -> None:
    def download(relative: str) -> None:
        _download_one(client, bucket, prefix + relative, local_root / relative)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(download, path) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="s3 download"):
            future.result()


def _write_marker(marker: Path, prefix: str, source: str) -> None:
    marker.write_text(
        json.dumps({"prefix": prefix, "source": source}, indent=2),
        encoding="utf-8",
    )


def _read_annotations(root: Path, split: str) -> dict[str, Any]:
    path = root / "annotations" / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет разметки для сплита {split}: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "classes" not in payload or "images" not in payload:
        raise ValueError(f"{path}: ожидались ключи 'classes' и 'images'")
    return payload


def _load_images(root: Path, records: list[dict[str, Any]], split: str) -> npt.NDArray[np.uint8]:
    images = np.zeros((len(records), *IMAGE_SHAPE), dtype=np.uint8)
    for index, record in enumerate(tqdm(records, desc=f"load {split}", leave=False)):
        path = root / str(record["file_name"])
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != IMAGE_SHAPE:
            raise ValueError(f"{path}: ожидалось {IMAGE_SHAPE}, получено {array.shape}")
        images[index] = array
    return images
