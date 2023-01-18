import warnings

import xgboost as xgb
from data.dataset import load_train_dataset
from evaluation.evaluate import rmsle_xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_x, train_y, test_size=0.2, random_state=0
    )

    dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(x_valid, y_valid, enable_categorical=True)
    watchlist = [(dtrain, "train"), (dvalid, "eval")]

    model = xgb.train(
        dict(cfg.model.params),
        dtrain=dtrain,
        evals=watchlist,
        feval=rmsle_xgb,
        num_boost_round=cfg.model.num_boost_round,
        verbose_eval=cfg.model.verbose_eval,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        maximize=False,
    )
    model.save_model(cfg.model.path + cfg.model.name)


if __name__ == "__main__":
    config = OmegaConf.load("../bike-sharing/config/xgboost.yaml")
    _main(config)
