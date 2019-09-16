import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import itertools
import numpy as np

from pylab import rcParams


data = pd.read_csv("/Users/ilyaperepelitsa/quant/dash_deploy/data/resampled/final_daily.csv")
data['Datetime'] = pd.to_datetime(data["Datetime"])
data = data.set_index('Datetime')
