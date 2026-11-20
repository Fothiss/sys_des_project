"""Фабрика моделей: baseline-линейка, своя CNN, архитектуры torchvision."""

from __future__ import annotations

from typing import Any, cast

from loguru import logger
import torch
from torch import nn
import torchvision

from sys_des_project.training.config import ConfigError, ModelConfig

DEFAULT_CHANNELS = [32, 64]


class LinearBaseline(nn.Module):
    """Baseline: один линейный слой поверх пикселей 3x32x32."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(self.flatten(x)))


class SimpleCNN(nn.Module):
    """Три свёрточных слоя с BN, два пулинга, GAP и линейный классификатор."""

    def __init__(self, channels: list[int], num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        first, second = channels
        self.features = nn.Sequential(
            _conv_block(3, first),
            _conv_block(first, first),
            nn.MaxPool2d(2),
            _conv_block(first, second),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(second, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.features(x)))


def build_model(model_cfg: ModelConfig, num_classes: int) -> nn.Module:
    """Собирает модель по секции model конфига."""
    model_type = model_cfg.type
    if model_type == "linear":
        model: nn.Module = LinearBaseline(num_classes)
    elif model_type == "simple_cnn":
        channels = _channels(model_cfg.params)
        dropout = float(model_cfg.params.get("dropout", 0.0) or 0.0)
        if not 0.0 <= dropout < 1.0:
            raise ConfigError("model.params.dropout: должно быть в [0, 1)")
        model = SimpleCNN(channels, num_classes, dropout=dropout)
    elif hasattr(torchvision.models, model_type):
        factory = getattr(torchvision.models, model_type)
        model = factory(weights=None, num_classes=num_classes)
        logger.info(f"torchvision.{model_type}: без предобученных весов")
    else:
        raise ConfigError(
            f"model.type={model_type!r}: поддерживаются 'linear', 'simple_cnn' "
            "или имя архитектуры из torchvision"
        )

    logger.info(f"Модель {model_type}: {count_parameters(model):,} обучаемых параметров")
    return model


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров — для лога и README."""
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _channels(params: dict[str, Any]) -> list[int]:
    raw = params.get("channels", DEFAULT_CHANNELS)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(
            f"model.params.channels: ожидался список из двух чисел, получено {raw!r}"
        )
    channels = [int(value) for value in raw]
    if any(channel <= 0 for channel in channels):
        raise ConfigError(
            f"model.params.channels: все значения должны быть > 0, получено {channels}"
        )
    return channels
