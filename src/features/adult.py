from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features to the dataset.
    Args:
        df (pd.DataFrame): Dataset.
    Returns:
        pd.DataFrame: Dataset with new features.
    """
    df["fnlwgt"] = np.log1p(df["fnlwgt"])
    df["workclass_occupation"] = df["workclass"] + "#" + df["occupation"]
    df["workclass_education"] = df["workclass"] + "#" + df["education"]
    df["occupation_education"] = df["occupation"] + "#" + df["education"]
    df["marital_status_relationship"] = df["marital_status"] + "#" + df["relationship"]
    df["race_sex"] = df["race"] + "#" + df["sex"]
    df["capital_margin"] = df["capital_gain"] - df["capital_loss"]
    df["capital_total"] = df["capital_gain"] + df["capital_loss"]
    df["capital_margin_flag"] = np.nan
    df.loc[df["capital_margin"] == 0, "capital_margin_flag"] = "zero"
    df.loc[df["capital_margin"] > 0, "capital_margin_flag"] = "positive"
    df.loc[df["capital_margin"] < 0, "capital_margin_flag"] = "negative"

    return df


def encode_frequency(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Frequency encoding for categorical features.
    Args:
        df (pd.DataFrame): Dataset.
        columns (list): List of categorical features.
    Returns:
        pd.DataFrame: Dataset with frequency encoded features.
    """
    for column in columns:
        vc = df[column].value_counts(dropna=False, normalize=True).to_dict()
        df[f"{column}"] = df[column].map(vc)
        df[f"{column}"] = df[f"{column}"].astype("float32")

    return df


def categorize_train_features(config: DictConfig, train: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        train: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder
    label_encoder = LabelEncoder()

    for cat_feature in tqdm(config.data.cat_features):
        train[cat_feature] = label_encoder.fit_transform(train[cat_feature])
        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

    return train


def categorize_test_features(config: DictConfig, test: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        config: config
        test: dataframe
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.data.encoder

    for cat_feature in tqdm(config.data.cat_features):
        le_encoder = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        test[cat_feature] = le_encoder.transform(test[cat_feature])

    return test
