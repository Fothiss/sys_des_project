import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# CIFAR-10
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
S3_RAW_KEY = f"raw/{CIFAR_ARCHIVE_NAME}"
S3_PROCESSED_PREFIX = "processed/cifar10"

# RustFS / S3
RUSTFS_ACCESS_KEY = os.getenv("RUSTFS_ACCESS_KEY")
RUSTFS_SECRET_KEY = os.getenv("RUSTFS_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_BUCKET = os.getenv("S3_BUCKET", "cifar10")

if not RUSTFS_ACCESS_KEY or not RUSTFS_SECRET_KEY:
    raise RuntimeError(
        "Не найдены RUSTFS_ACCESS_KEY / RUSTFS_SECRET_KEY. "
        "Проверьте .env или переменные окружения."
    )
