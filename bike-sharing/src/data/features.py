from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add features to.

    Returns:
        pd.DataFrame: The dataframe with added features.
    """
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.weekofyear
    df["quarter"] = df["datetime"].dt.quarter

    return df


def delete_outliers(train_y: pd.Series, train_x: pd.DataFrame) -> pd.DataFrame:
    """Delete outliers from the training dataset.
    Args:
        train_y (pd.Series): The target.
        train_x (pd.DataFrame): The training dataset.

    Returns:
        pd.DataFrame: The training dataset without outliers.
    """
    original_shape = train_x.shape
    mean = np.mean(train_y)
    std = np.std(train_y)
    outliers = np.abs(train_y - mean) > (3 * std)
    outliers_num = len(train_x[outliers])
    train_x = train_x.drop(index=train_y[outliers])
    print("Have already deleted", outliers_num, "outliers")
    print("Shape Before Delete Ouliers: ", original_shape)
    print("Shape After Delete Ouliers: ", train_x.shape)

    return train_x


def predict_wind_speed(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the wind speed using the wind direction.
    Args:
        df (pd.DataFrame): The dataframe to predict the wind speed.

    Returns:
        pd.DataFrame: The dataframe with predicted wind speed.
    """
    df = pd.concat([train_x, test_x])
    df_without_wind = df[df["windspeed"] == 0]
    df_with_wind = df[df["windspeed"] != 0]

    rf = RandomForestRegressor()
    wind_columns = ["season", "weather", "humidity", "month", "temp", "year"]
    rf.fit(df_with_wind[wind_columns], df_with_wind["windspeed"])
    wind_preds = rf.predict(X=df_without_wind[wind_columns])
    df.loc[df["windspeed"] == 0, "windspeed"] = wind_preds

    train_x = df[: train_x.shape[0]]
    test_x = df[train_x.shape[0] :]

    return train_x, test_x
