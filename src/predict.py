from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.adult import Dataset
from models.infer import inference, load_model


@hydra.main(config_path="../config/", config_name="predict", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.path

    # data load
    data_loader = Dataset(config=cfg)

    test_x = data_loader.load_test_dataset()
    submit = data_loader.load_submit_dataset()

    # model load
    results = load_model(cfg, cfg.models.results)

    # infer test
    preds = inference(results, test_x)
    preds = preds if cfg.output.predict_proba else np.where(preds > 0.5, 1, 0)

    submit[cfg.output.target] = preds
    submit.to_csv(path / cfg.models.output, index=False)


if __name__ == "__main__":
    _main()
