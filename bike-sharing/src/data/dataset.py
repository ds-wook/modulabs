from typing import Tuple

import numpy as np
import pandas as pd
from data.features import build_features
from omegaconf import DictConfig


def load_train_dataset(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the training dataset.
    Args:
        config (DictConfig): The configuration.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: The training dataset and the target.
    """
    train = pd.read_csv(config.data.path + config.data.train)
    train["datetime"] = pd.to_datetime(train["datetime"])

    train = build_features(train)
    train_x = train.drop(columns=[*config.data.del_features, config.data.target])
    train_y = train[config.data.target]
    train_y = np.log1p(train_y)

    return train_x, train_y


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    """Load the test dataset.
    Args:
        config (DictConfig): The configuration.
    Returns:
        pd.DataFrame: The test dataset.
    """
    test = pd.read_csv(config.data.path + config.data.test)
    test["datetime"] = pd.to_datetime(test["datetime"])

    test = build_features(test)
    test_x = test.drop(columns=["datetime"])

    return test_x
