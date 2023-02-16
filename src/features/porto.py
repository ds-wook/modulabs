import pandas as pd
from omegaconf import DictConfig
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder


def add_features(cfg: DictConfig, data: pd.DataFrame) -> pd.DataFrame:
    data["num_missing"] = (data == -1).sum(axis=1)
    all_features = data.columns.tolist()
    remaining_features = [col for col in all_features if "cat" not in col and "calc" not in col] + ["num_missing"]
    cat_features = [col for col in all_features if "cat" in col]  # Nominal features

    # Apply One-Hot encoding
    onehot_encoder = OneHotEncoder()
    encoded_cat_matrix = onehot_encoder.fit_transform(data[cat_features])

    # Feature with 'ind' on tag
    ind_features = [col for col in all_features if "ind" in col]

    first_col = True
    for col in ind_features:
        if first_col:
            data["mix_ind"] = data[col].astype(str) + "_"
            first_col = False
        else:
            data["mix_ind"] += data[col].astype(str) + "_"

    cat_count_features = []
    for col in cat_features + ["mix_ind"]:
        val_counts_dic = data[col].value_counts().to_dict()
        data[col + "_count"] = data[col].apply(lambda x: val_counts_dic[x])
        cat_count_features.append(f"{col}_count")

    all_data_remaining = data[remaining_features + cat_count_features].drop(cfg.data.drop_features)
    all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data_remaining), encoded_cat_matrix], format="csr")

    return all_data_sprs
