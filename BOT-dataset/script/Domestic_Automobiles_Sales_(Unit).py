# %%
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

# %%
df = pd.read_csv("data-raw/EC_EI_001_S2_ENG_ALL.CSV", index_col=0)

df.columns = df.columns.str[:-2]
df.columns = pd.to_datetime(df.columns, format="%b %Y", errors="coerce")

df.index = df.index.str.strip()
df = df.loc["Domestic Automobiles Sales (Unit)"]
df.index.name = "date"

# %%
# Save to Local
filename = "Domestic_Automobiles_Sales_(Unit).csv"
df.to_csv("data-prep/" + filename)
