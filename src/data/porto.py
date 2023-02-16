from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from scipy import sparse

from features.porto import add_features


class PortoSeguroDataset:
    def __init__(self, config: DictConfig):
        self.config = config
        self.path = Path(self.config.data.path)
        self._train = pd.read_csv(self.path / self.config.data.train)
        self._test = pd.read_csv(self.path / self.config.data.test)
        self._submit = pd.read_csv(self.path / self.config.data.submit)

    def load_dataset(self) -> tuple[sparse.hstack, pd.Series, sparse.hstack]:
        train_x = self._train.drop(columns=[*self.config.data.drop_features, self.config.data.target])
        test_x = self._test.drop(columns=[*self.config.data.drop_features])
        train_y = self._train[self.config.data.target]
        all_data = pd.concat([train_x, test_x], ignore_index=True)
        all_data = add_features(self.config, all_data)
        train_x = all_data[: train_x.shape[0]]
        test_x = all_data[train_x.shape[0] :]

        return train_x, train_y, test_x

    def load_submit_dataset(self) -> pd.DataFrame:
        return self._submit
