"""Загрузка сырого архива CIFAR-10 в S3 (идемпотентно, без перезаписи)."""

import os
from pathlib import Path
import urllib.request

import typer

from sys_des_project.config import CIFAR_ARCHIVE_NAME, CIFAR_URL, RAW_DATA_DIR, S3_RAW_KEY
from sys_des_project.s3 import get_s3_client, object_exists

app = typer.Typer()


def download_archive(dest: Path) -> None:
    """Скачивает архив CIFAR-10, если его ещё нет локально."""
    if dest.exists():
        print(f"Архив уже есть локально: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {CIFAR_URL} ...")
    urllib.request.urlretrieve(CIFAR_URL, dest)
    print(f"Скачано: {dest} ({dest.stat().st_size} байт)")


@app.command()
def main(
    local_dir: Path = typer.Option(RAW_DATA_DIR, help="Куда временно скачать архив"),
) -> None:
    """Загружает cifar-10-python.tar.gz в s3://<bucket>/raw/ без изменений."""
    bucket = os.environ["S3_BUCKET"]
    s3 = get_s3_client()

    if object_exists(s3, bucket, S3_RAW_KEY):
        print(f"Объект уже в S3: s3://{bucket}/{S3_RAW_KEY}. Пропускаю.")
        raise typer.Exit(code=0)

    local_path = local_dir / CIFAR_ARCHIVE_NAME
    download_archive(local_path)

    if object_exists(s3, bucket, S3_RAW_KEY):
        print("Объект появился в S3 параллельно. Пропускаю загрузку.")
        raise typer.Exit(code=0)

    print(f"Загружаю {local_path} -> s3://{bucket}/{S3_RAW_KEY} ...")
    s3.upload_file(str(local_path), bucket, S3_RAW_KEY)
    print("Готово.")


if __name__ == "__main__":
    app()
