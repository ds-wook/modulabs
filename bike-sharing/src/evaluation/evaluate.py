from typing import Tuple

import numpy as np
from xgboost import DMatrix


def rmsle(y_true: np.ndarray, y_pred: np.ndarray, convert_exp: bool = True) -> float:
    """Calculate the RMSLE.
    Args:
        y_true (np.ndarray): The true values.
        y_preds (np.ndarray): The predicted values.
        convert_exp (bool, optional): Convert the values to exponential.
    Returns:
        float: The RMSLE.
    """
    if convert_exp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    # 로그변환 후 결측값을 0으로 변환
    log_true = np.nan_to_num(np.log(y_true + 1))
    log_pred = np.nan_to_num(np.log(y_pred + 1))

    # RMSLE 계산
    output = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return output


def rmsle_xgb(predictions: np.ndarray, dmat: DMatrix) -> Tuple[str, float]:
    labels = dmat.get_label()
    diffs = np.log(predictions + 1) - np.log(labels + 1)
    squared_diffs = np.square(diffs)
    avg = np.mean(squared_diffs)
    return "rmsle", np.sqrt(avg)
