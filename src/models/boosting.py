from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from models.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _evaluate_metric(self, y_hat: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return "f1", f1_score(y_true, y_hat, average="micro"), True

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> lgb.Booster:
        """
        load train model
        """
        train_set = lgb.Dataset(data=X_train, label=y_train, categorical_feature=[*self.config.data.cat_features])
        valid_set = lgb.Dataset(data=X_valid, label=y_valid, categorical_feature=[*self.config.data.cat_features])

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.models.params),
            num_boost_round=self.config.models.num_boost_round,
            feval=self._evaluate_metric,
            callbacks=[
                lgb.log_evaluation(self.config.models.verbose_eval),
                lgb.early_stopping(self.config.models.early_stopping_rounds),
            ],
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> CatBoostClassifier:
        """
        load train model
        """
        train_data = Pool(data=X_train, label=y_train, cat_features=[*self.config.data.cat_features])
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=[*self.config.data.cat_features])

        model = CatBoostClassifier(
            random_state=self.config.models.seed,
            cat_features=self.config.data.cat_features,
            **self.config.models.params,
        )
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose=self.config.models.verbose,
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _evaluate_metric(self, y_hat: np.ndarray, data: xgb.DMatrix) -> tuple[str, float]:
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return "f1", 1 - f1_score(y_true, y_hat, average="micro")

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> xgb.Booster:
        """
        load train model
        """
        dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dict(self.config.models.params),
            dtrain=dtrain,
            evals=watchlist,
            feval=self._evaluate_metric,
            maximize=True,
            num_boost_round=self.config.models.num_boost_round,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose_eval=self.config.models.verbose_eval,
        )

        return model
