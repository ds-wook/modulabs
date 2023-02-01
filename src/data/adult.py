from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from features.adult import add_features, categorize_test_features, categorize_train_features, encode_frequency


class AdultDataset:
    def __init__(self, config: DictConfig):
        self.config = config
        self.path = Path(self.config.data.path)
        self._train = pd.read_csv(self.path / self.config.data.train, na_values="?")
        self._test = pd.read_csv(self.path / self.config.data.test, na_values="?")
        self._submit = pd.read_csv(self.path / self.config.data.submit)

    def load_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        self._train = add_features(self._train)
        self._train = categorize_train_features(self.config, self._train)
        self._train = encode_frequency(self._train, self.config.data.freq_features)
        train_x = self._train.drop(columns=[*self.config.data.drop_features, self.config.data.target])
        train_y = self._train[self.config.data.target].map({"<=50K": 0, ">50K": 1})

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        self._test = add_features(self._test)
        self._test = categorize_test_features(self.config, self._test)
        self._test = encode_frequency(self._test, self.config.data.freq_features)
        test_x = self._test.drop(columns=[*self.config.data.drop_features])

        return test_x

    def load_submit_dataset(self) -> pd.DataFrame:
        return self._submit

    def load_target_dataset(self) -> pd.Series:
        return self._train[self.config.data.target].map({"<=50K": 0, ">50K": 1})
