"""Метрики на val/test: loss, accuracy, macro-F1 и матрица ошибок."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score
import torch


@dataclass
class EvalResult:
    """Метрики одного прохода по датасету."""

    loss: float
    accuracy: float
    macro_f1: float
    targets: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        """Только скалярные метрики — для логгера."""
        return {"loss": self.loss, "accuracy": self.accuracy, "macro_f1": self.macro_f1}


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: torch.nn.Module,
) -> EvalResult:
    """Считает loss/accuracy/macro-F1; предсказания запоминает для матрицы ошибок."""
    model.eval()
    total_loss = 0.0
    total = 0
    targets: list[int] = []
    preds: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            batch_size = int(inputs.size(0))
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            preds.extend(int(value) for value in logits.argmax(dim=1).tolist())
            targets.extend(int(value) for value in labels.tolist())

    if total == 0:
        raise RuntimeError("Пустой датасет: оценивать нечего")

    correct = sum(1 for pred, target in zip(preds, targets, strict=True) if pred == target)
    return EvalResult(
        loss=total_loss / total,
        accuracy=correct / total,
        macro_f1=float(f1_score(targets, preds, average="macro", zero_division=0.0)),
        targets=targets,
        preds=preds,
    )


def confusion_matrix_list(
    targets: list[int], preds: list[int], num_classes: int
) -> list[list[int]]:
    """Матрица ошибок как вложенные списки — кладём в metrics.json."""
    matrix = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    return [[int(value) for value in row] for row in matrix.tolist()]


def confusion_figure(targets: list[int], preds: list[int], classes: list[str]) -> Any:
    """Матрица ошибок как matplotlib Figure (backend Agg, без GUI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(matrix, cmap="Blues")
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    ticks = list(range(len(classes)))
    axes.set_xticks(ticks, labels=classes, rotation=45, ha="right")
    axes.set_yticks(ticks, labels=classes)

    threshold = matrix.max() / 2 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axes.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    return figure
