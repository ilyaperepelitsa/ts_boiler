import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import itertools
import numpy as np

from pylab import rcParams

data = pd.read_csv("/Users/ilyaperepelitsa/quant/dash_deploy/data/resampled/final_daily.csv")
data['Datetime'] = pd.to_datetime(data["Datetime"])
data = data.set_index('Datetime')
# data.drop(['Datetime'], axis=1, inplace=True)
# data.head()

rcParams['figure.figsize'] = 30, 10
decomposition = sm.tsa.seasonal_decompose(data['Global_active_power'], model='additive')
fig = decomposition.plot()
dir(decomposition)

decomposition.observed
decomposition.trend.tail()
decomposition.seasonal.tail()
decomposition.resid.tail()






cycle, trend = sm.tsa.filters.hpfilter(series, 50)

from statsmodels.tsa.seasonal import STL



result = STL(series).fit()
chart = result.plot()
plt.show()



import datetime
# Then you'll have, using datetime.timedelta:

date_1 = datetime.datetime.strptime(start_date, "%m/%d/%y")

end_date = date_1 + datetime.timedelta(days=10)
