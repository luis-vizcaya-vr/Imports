import pandas as pd
from darts import *
from darts import metrics 
import matplotlib.pyplot as plt
import darts as drt
from darts.models import *
from darts.utils.utils import *

FORECAST_PERIOD = 14
df = pd.read_csv('NATGAS-DATA_v2.csv', usecols= [0,1])
print('Longitud antes: ',len(df))
split_date = pd.datetime(2021,12,31)
df['Date1'] = pd.date_range(start='1/1/2020', end = '03/01/2022', freq='D')
df = df.loc[df['Date1']< split_date]

print('Longitud despues:',len(df))
series = TimeSeries.from_dataframe(df, time_col = 'Date1',  value_cols ='NatGas - Dominion South', freq = 'd', fill_missing_dates = True)
train, val = series[:-FORECAST_PERIOD], series[-FORECAST_PERIOD:]

def eval_model(model,train):
  res = {}
  print('Running: ', model)
  model.fit(train)
  forecast = model.predict(len(val))
  res['MAPE'] = drt.metrics.metrics.mape(val, forecast)
  res['MASE'] = drt.metrics.metrics.mase(val, forecast,train)
  res['R2'] = drt.metrics.metrics.r2_score(val, forecast)
  res['Forecast'] = forecast
  
  backtest = model.historical_forecasts(series=train, 
                                          start=0.8, 
                                          verbose=True, 
                                          forecast_horizon=FORECAST_PERIOD )
  #print('Backtest: ', backtest)  
  return res

def Best_model(model_list, train, metric = 'MAPE'):
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
        {"trend": [ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE]},
    FFT:
        {"nr_freqs_to_keep": [2,3,4,5,6,7,8],
        "trend":['poly', 'exp', 'None'],
        "trend_poly_degree": [1,2,3,]
         },
    KalmanForecaster:
        {"dim_x": [2,3,4,5]}

}

'''
Exponential = ExponentialSmoothing.gridsearch(parameters = parameters[ExponentialSmoothing], series = series, forecast_horizon= FORECAST_PERIOD, metric = metrics.metrics.mape)
Fourier = FFT.gridsearch(parameters = parameters[FFT], series = series, forecast_horizon= FORECAST_PERIOD)
Kalman = KalmanForecaster.gridsearch(parameters = parameters[KalmanForecaster], series = series, forecast_horizon= FORECAST_PERIOD)
model_list = [AutoARIMA(), Exponential[0], Fourier[0], Kalman[0]]
'''
model_list = [AutoARIMA(), NaiveDrift()]


a = Best_model(model_list, train, metric = 'MASE')


