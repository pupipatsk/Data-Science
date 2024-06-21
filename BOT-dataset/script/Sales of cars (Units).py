# %%
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

# %%
# BOT format
filepath = "data-raw/EC_EI_003_S2_ENG_ALL.CSV"

df = pd.read_csv(filepath, index_col=0, encoding="utf-8")

# columns datetime
df.columns = df.columns.str[:-2]  # remove r(revised), p(preliminary)
df.columns = pd.to_datetime(df.columns, format="%b %Y", errors="coerce")
df.columns = df.columns + pd.offsets.MonthEnd()

# index features
df.index = df.index.str.strip()
features = [
    "Sales of Commercial cars (Units)",
    "Sales of Passenger cars (Units)",
    "Sales of Commercial cars (Units) (Seasonally Adjusted)",
    "Sales of Passenger cars (Units) (Seasonally Adjusted)",
]
df = df[df.index.isin(features)]

df = df.transpose()
df.index.name = "date"

df

# %%
# Save to Local
filename = "Sales of cars (Units).csv"

df.to_csv("data-prep/" + filename, encoding="utf-8")
