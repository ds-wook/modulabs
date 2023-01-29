from __future__ import annotations

import logging

import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import f1_score


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: dict[str, float]) -> None:
    """
    Evaluate metrics
    """

    oof_score = f1_score(y_true, np.round(y_pred), average="micro")

    tables = PrettyTable()
    tables.field_names = ["name", "scores"]

    for fold, score in scores.items():
        tables.add_row([fold, f"{score:.4f}"])

    tables.add_row(["OOF score", f"{oof_score:.4f}"])

    logging.info(f"\n{tables.get_string()}")
