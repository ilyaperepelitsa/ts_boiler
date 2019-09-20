import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import itertools
import numpy as np

from pylab import rcParams
from pyramid.arima import auto_arima

data = pd.read_csv("/Users/ilyaperepelitsa/quant/dash_deploy/data/resampled/final_daily.csv")
data['Datetime'] = pd.to_datetime(data["Datetime"])
data = data.set_index('Datetime')


# p = d = q = range(0, 20)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(data['Global_active_power'],
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

stepwise_model = auto_arima(data['Global_active_power'],
                           start_p=1, start_q=1,
                           max_p=50, max_q=50, m=12,
                           start_P=0,
                           start_Q =0,
                           seasonal=True,
                           d=1, D=1,
                           trace=True,
                           test = 'kpss',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False,
                           random = True,
                           n_fits = 50)
