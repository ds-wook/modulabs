# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
train = pd.read_csv("../input/adult-income-prediction/train.csv", na_values="?")
train.head()

# %%
sns.histplot(train["fnlwgt"]);

# %%
sns.histplot(np.log1p(train["fnlwgt"]));

# %%

