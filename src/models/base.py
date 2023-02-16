from __future__ import annotations

import gc
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from evaluation.evaluate import evaluate_metrics

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]
    scores: dict[str, dict[str, float]]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        self.config = config
        self.result = None

    def save_model(self, model_path: Path | str, model_name: str) -> BaseModel:
        """
        Save model
        Args:
            model_path: model path
            model_name: model name
        Return:
            Model Result
        """

        with open(model_path / model_name, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

        return self

    @abstractclassmethod
    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> NoReturn:
        """Trains the model"""
        raise NotImplementedError

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> BaseModel:
        """
        Trains the model.
        Args:
            X_train: train dataset
            y_train: target dataset
            X_valid: validation dataset
            y_valid: validation target dataset
        Return:
            Model Result
        """
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def cross_validation(self, train_x: pd.DataFrame, train_y: pd.Series) -> ModelResult:
        """
        Train data
        Args:
            train_x: train dataset
            train_y: target dataset
        Return:
            Model Result
        """
        models = dict()
        scores = dict()
        folds = self.config.models.n_splits
        seed = self.config.data.seed

        str_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = str_kf.split(train_x, train_y)
        oof_preds = np.zeros(len(train_x))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            # split train and validation data
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._fit(X_train, y_train, X_valid, y_valid)
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = (
                model.predict(X_valid)
                if isinstance(model, lgb.Booster)
                else model.predict(xgb.DMatrix(X_valid))
                if isinstance(model, xgb.Booster)
                else model.predict_proba(X_valid)[:, 1]
            )

            # score
            score = f1_score(y_valid, np.round(oof_preds[valid_idx]), average="micro")

            scores[f"Fold{fold}"] = score

            del X_train, X_valid, y_train, y_valid, model
            gc.collect()

        self.result = ModelResult(oof_preds=oof_preds, models=models, scores={"KFold_scores": scores})

        # evaluate
        evaluate_metrics(train_y, oof_preds, scores)

        return self.result
