import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.adult import Dataset

warnings.filterwarnings("ignore")


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.name
    data_loader = Dataset(config=cfg)
    submission = data_loader.load_submit_dataset()

    lgbm_preds = pd.read_csv(path / cfg.output.lgbm_preds)
    xgb_preds = pd.read_csv(path / cfg.output.xgb_preds)
    cb_preds = pd.read_csv(path / cfg.output.cb_preds)

    preds = np.array(
        [xgb_preds.prediction.to_numpy(), cb_preds.prediction.to_numpy(), lgbm_preds.prediction.to_numpy()]
    ).T

    columns = ["xgb", "cb", "lgbm"]
    ensemble = pd.DataFrame(preds, columns=columns)
    ensemble["voting_sum"] = ensemble.sum(axis=1)
    blending_preds = ensemble["voting_sum"].apply(lambda x: 1 if x >= 2 else 0).to_numpy()

    submission["prediction"] = blending_preds
    submission.to_csv(path / cfg.output.final_preds, index=False)


if __name__ == "__main__":
    _main()
