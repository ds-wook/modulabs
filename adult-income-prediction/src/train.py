from __future__ import annotations

import hydra
from omegaconf import DictConfig

from data.dataset import Dataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # data load
    data_loader = Dataset(config=cfg)
    train_x, train_y = data_loader.load_train_dataset()

    if cfg.models.name == "xgboost":
        # train model
        xgb_trainer = XGBoostTrainer(config=cfg)
        xgb_trainer.train_cross_validation(train_x, train_y)
        # save model
        xgb_trainer.save_model()

    elif cfg.models.name == "lightgbm":
        # train model
        lgb_trainer = LightGBMTrainer(config=cfg)
        lgb_trainer.train_cross_validation(train_x, train_y)
        # save model
        lgb_trainer.save_model()

    elif cfg.models.name == "catboost":
        # train model
        cat_trainer = CatBoostTrainer(config=cfg)
        cat_trainer.train_cross_validation(train_x, train_y)
        # save model
        cat_trainer.save_model()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
