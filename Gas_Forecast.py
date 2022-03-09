from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
)
from tabnanny import verbose
import pandas as pd
from darts import *
from darts import metrics 
import matplotlib.pyplot as plt
import darts as drt
from darts.models import *
from darts.utils.utils import *
from darts.dataprocessing.transformers import (Scaler, MissingValuesFiller, Mapper, InvertibleMapper,)
import winsound
import seaborn as sn





FORECAST_PERIOD = 14
temp = pd.read_csv('NATGAS-DATA_V3.csv', usecols= [0,11])

#corrMatrix = data.corr()
#sn.heatmap(corrMatrix, annot=True)
#plt.show()
df = pd.read_csv('NATGAS-DATA_V3.csv', usecols= [0,2])

End_Date = pd.datetime(2021,12,31)
Start_Date = pd.datetime(2021,5,1)
df['Date'] = pd.DatetimeIndex(df['Date'])
df = df.loc[(df['Date'] <= End_Date) & (df['Date'] >= Start_Date)]
temp['Date'] = pd.DatetimeIndex(temp['Date'])
temp = temp.loc[(temp['Date'] <= End_Date) & (temp['Date'] >= Start_Date)]


series = TimeSeries.from_dataframe(df, time_col = 'Date', freq = 'd', fill_missing_dates = True)
#scaler = Scaler()
#series = scaler.fit_transform(series)

series.plot(label = 'real price')
#plt.show()


temp_serie = TimeSeries.from_dataframe(temp, time_col = 'Date', freq = 'd', fill_missing_dates = True)
#series.plot(label = 'temperature')

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
    "ARIMA":
        {"p": [1],
        "q": [1],
        "d": [0]
       },

    "ExponentialSmoothing":
        {"trend": [ModelMode.NONE,ModelMode.MULTIPLICATIVE]},

    "T":
        {
          "theta":[0.5, 1],
        },
        
    "VARIMA":
        {"p": [1],
        "q": [1],
        "d": [0],
        },

    "KALMAN":
        {"dim_x": [2,3,4],
        },
    } 

#print("Gridsearch Exponential..")

Tet = FourTheta.gridsearch(parameters = parameters["T"], series = series, forecast_horizon = FORECAST_PERIOD, metric = metrics.metrics.mape, n_jobs = -1 ,  future_covariates=temp_serie)
Arima_model = ARIMA.gridsearch(parameters = parameters["ARIMA"], forecast_horizon = FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
Expon = ExponentialSmoothing.gridsearch(parameters = parameters["ExponentialSmoothing"], forecast_horizon= FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
#Auto_Arima = AutoARIMA()
#Auto_Arima.fit(series = series,  future_covariates = temp_serie)

#Varima = VARIMA.gridsearch(parameters = parameters["VARIMA"], forecast_horizon = FORECAST_PERIOD, series = series)
#Kalman = KalmanForecaster.gridsearch(series = series,parameters = parameters["KALMAN"],  forecast_horizon = FORECAST_PERIOD,  future_covariates=temp_serie)
#a = Best_model(model_list, train, metric = 'MAPE')
#print('Historical Forecast..')

#h = Kalman[0].historical_forecasts(series = series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
#mape = metrics.metrics.mape(h, series)
#h.plot(label = 'Arima: '+ str(mape))

h = Tet[0].historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
mape = metrics.metrics.mape(h, series)
h.plot(label = 'Theta: '+ str(mape))

h = Expon[0].historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
mape = metrics.metrics.mape(h, series)
h.plot(label = 'Exponential: '+ str(mape))

#h = Auto_Arima.historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
#mape = metrics.metrics.mape(h, series)
#h.plot(label = 'AutoArima: '+ str(mape))

h = Arima_model[0].historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
mape = metrics.metrics.mape(h, series)
h.plot(label = 'Arima: '+ str(mape))


'''
h = Varima[0].historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
mape = metrics.metrics.mape(h, series)
h.plot(label = 'Varima: '+ str(mape))

#series.to_csv("serie.csv")
#h.to_csv("gas_price.csv")
'''
winsound.Beep(440, 100)
plt.ylim(0,7)
plt.show()
plt.savefig('model.jpg')




#h = Kalman[0].historical_forecasts(series, start = 0.4, forecast_horizon= FORECAST_PERIOD)
#h = Fourier[0].historical_forecasts(series, start = 0.4, forecast_horizon= FORECAST_PERIOD)
#print(metrics.metrics.mape(h, series))'''
