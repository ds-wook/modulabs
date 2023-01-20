# %%
import pandas as pd

# %%
train = pd.read_csv("../input/adult-income-prediction/train.csv", na_values="?")
train.head()
# %%
train["capital_gain"].value_counts()
# %%
