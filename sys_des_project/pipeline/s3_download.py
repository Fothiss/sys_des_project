"""Скачивание сырого архива CIFAR-10 из S3."""

import hashlib
from pathlib import Path

import boto3
from loguru import logger
import typer

DEFAULT_BUCKET = "cifar10"
DEFAULT_KEY = "raw/cifar-10-python.tar.gz"
DEFAULT_DEST = Path("data/raw/cifar-10-python.tar.gz")
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(
    bucket: str = typer.Option(DEFAULT_BUCKET, help="S3 bucket"),
    key: str = typer.Option(DEFAULT_KEY, help="S3 key"),
    dest: Path = typer.Option(DEFAULT_DEST, help="Локальный путь"),
    force: bool = typer.Option(False, help="Перекачать принудительно"),
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        actual = md5sum(dest)
        if actual == EXPECTED_MD5:
            logger.info(f"{dest} уже есть, md5 совпадает — пропускаю")
            return
        logger.warning(f"md5 {actual} != {EXPECTED_MD5}, перекачиваю")

    logger.info(f"Скачиваю s3://{bucket}/{key} -> {dest}")
    boto3.client("s3").download_file(bucket, key, str(dest))
    logger.info("Готово")


if __name__ == "__main__":
    typer.run(main)
