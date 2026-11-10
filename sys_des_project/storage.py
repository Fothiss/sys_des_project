# sys_des_project/storage.py
import boto3
from botocore.client import BaseClient
from botocore.config import Config

from sys_des_project.config import (
    RUSTFS_ACCESS_KEY,
    RUSTFS_SECRET_KEY,
    S3_ENDPOINT,
)


def get_s3_client() -> BaseClient:
    """Создаёт boto3-клиент для RustFS с явными credentials и endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=RUSTFS_ACCESS_KEY,
        aws_secret_access_key=RUSTFS_SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
