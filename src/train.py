from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from data.adult import AdultDataset
from models.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # data load
    data_loader = AdultDataset(config=cfg)
    train_x, train_y = data_loader.load_train_dataset()

    if cfg.models.name == "xgboost":
        # train model
        xgb_trainer = XGBoostTrainer(config=cfg)
        xgb_trainer.cross_validation(train_x, train_y)
        # save model
        xgb_trainer.save_model(Path(cfg.models.path), cfg.models.results)

    elif cfg.models.name == "lightgbm":
        # train model
        lgb_trainer = LightGBMTrainer(config=cfg)
        lgb_trainer.cross_validation(train_x, train_y)
        # save model
        lgb_trainer.save_model(Path(cfg.models.path), cfg.models.results)

    elif cfg.models.name == "catboost":
        # train model
        cb_trainer = CatBoostTrainer(config=cfg)
        cb_trainer.cross_validation(train_x, train_y)
        # save model
        cb_trainer.save_model(Path(cfg.models.path), cfg.models.results)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
