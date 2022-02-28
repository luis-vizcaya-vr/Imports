import pandas as pd
from darts import *
import matplotlib.pyplot as plt
import darts as drt
FORECAST_PERIOD = 14
from darts.models import *
from darts.utils.utils import *
#from metatrader5 import *

df = pd.read_csv('NATGAS-DATA.csv', usecols= [0,1])
df['Date1'] = pd.date_range(start='9/1/2021', end = '02/21/2022', freq='D')

series = TimeSeries.from_dataframe(df, time_col = 'Date1',  value_cols ='NatGas - Transco Z6 Non-NY', freq = 'd', fill_missing_dates = True)
train, val = series[:-FORECAST_PERIOD], series[-FORECAST_PERIOD:]

def eval_model(model,train):
  res = {}
  print('Running: ', model)
  model.fit(train)
  forecast = model.predict(len(val))
  res['MAPE'] = drt.metrics.metrics.mape(val, forecast)
  res['R2'] = drt.metrics.metrics.r2_score(val, forecast)
  res['Forecast'] = forecast
  return res

def Best_model(model_list, train,metric = 'MAPE'):
  Best = ''
  Plot_Id = 100
  results = {}
  Max_Dev = float("Inf")
  for m in model_list:
    r = eval_model(m,train)
    Plot_Id += 1
    print(metric,' :', r[metric])
    if r[metric] < Max_Dev:
      Max_Dev = r[metric]
      results = r
      results['Model'] = str(m)
  return results



parameters = {
    ARIMA:
        {"p": [1,2,3],
        "q": [1]},
    ExponentialSmoothing:
        {"trend": [ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE, 'None']},
    FFT:
        {"nr_freqs_to_keep": [2,3,4,5,6,7,8,9,10,11,12,13,14],
        "trend":['poly', 'exp', 'None'],
        "trend_poly_degree": [1,2,3,4]
         },
    KalmanForecaster:
        {"dim_x": [2,3,4,5,6]}

}


Exponential = ExponentialSmoothing.gridsearch(parameters = parameters[ExponentialSmoothing], series = series, forecast_horizon= FORECAST_PERIOD)
Fourier = FFT.gridsearch(parameters = parameters[FFT], series = series, forecast_horizon= FORECAST_PERIOD)
Kalman = KalmanForecaster.gridsearch(parameters = parameters[KalmanForecaster], series = series, forecast_horizon= FORECAST_PERIOD)
model_list = [AutoARIMA(), Exponential[0], Fourier[0], Kalman[0]]
a = Best_model(model_list, train, metric = 'MAPE')