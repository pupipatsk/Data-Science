### --- System and Path --- ###
import os
import sys

REPO_PATH = os.path.abspath(os.path.join('..')) # depend on specific directory structure
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

import time
import warnings
warnings.filterwarnings('ignore')


### --- Data Manipulation --- ###
import pandas as pd
import numpy as np
np.random.seed(42)
from tqdm import tqdm # Progress bar


### --- Modelling --- ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from xgboost import XGBRegressor


### --- Visualization --- ###
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams["font.family"] = "tahoma" # TH font

import seaborn as sns
sns.set_theme(style="whitegrid")

import plotly.express as px
import plotly.graph_objects as go


### --- iPython Config --- ###
from IPython import get_ipython
if 'IPython.extensions.autoreload' not in get_ipython().extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')
else:
    get_ipython().run_line_magic('reload_ext', 'autoreload')
%autoreload 2

### --- Custom Modules --- ###
from src import *
