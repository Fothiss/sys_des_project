"""Общий клиент boto3 для работы с локальным S3 (RustFS)."""

import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def get_s3_client() -> boto3.client:
    """Клиент boto3 с настройками для RustFS (path-style, s3v4)."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUSTFS_SECRET_KEY"],
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
        region_name="us-east-1",
    )


def object_exists(s3: boto3.client, bucket: str, key: str) -> bool:
    """Проверяет наличие объекта через head_object."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise
