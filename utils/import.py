import warnings
warnings.filterwarnings('ignore')
import time
import os

import pandas as pd
import numpy as np
np.random.seed(42)

# --- Modelling --- #
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from xgboost import XGBRegressor

# --- Visualization --- #
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams["font.family"] = "tahoma" # TH font
import seaborn as sns
sns.set_theme(style="whitegrid")
import plotly.express as px
