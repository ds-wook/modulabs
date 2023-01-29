from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


class PortoSeguroDataset:
    def __init__(self, config: DictConfig):
        self.config = config
        self.path = Path(self.config.data.path)
        self._train = pd.read_csv(self.path / self.config.data.train)
        self._test = pd.read_csv(self.path / self.config.data.test)
        self._submit = pd.read_csv(self.path / self.config.data.submit)

    def load_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        train_x = self._train.drop(columns=[*self.config.data.drop_features, self.config.data.target])
        train_y = self._train[self.config.data.target]
        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        test_x = self._test.drop(columns=[*self.config.data.drop_features])
        return test_x

    def load_submit_dataset(self) -> pd.DataFrame:
        return self._submit
