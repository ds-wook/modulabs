import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_test_dataset

warnings.filterwarnings("ignore")


def _main(cfg: DictConfig):
    test_x = load_test_dataset(cfg)
    submission = pd.read_csv(cfg.data.path + cfg.data.submission)
    model = xgb.Booster(model_file=cfg.model.path + cfg.model.name)
    preds = model.predict(xgb.DMatrix(test_x))
    submission[cfg.data.target] = np.expm1(preds)
    submission.to_csv(cfg.output.path + cfg.output.submission, index=False)


if __name__ == "__main__":
    config = OmegaConf.load("../bike-sharing/config/xgboost.yaml")
    _main(config)
