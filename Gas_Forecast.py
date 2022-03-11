from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
)
#from tabnanny import verbose
from datetime import date, timedelta
import datetime
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

START_POINT = 0.90
FORECAST_PERIOD = 14
Today = datetime.date.today()
Yesterday = Today - timedelta(days = 1)
Forecast_Date = Today + timedelta(days = FORECAST_PERIOD)
Fuel_list = list(pd.read_csv('FUEL_SIDS.csv')['FUEL'])
Fuel_list = Fuel_list[0:2]

temp = pd.read_csv('NATGAS-DATA_V3.csv', usecols= [0,11])
fuel_mapping = pd.read_csv('fuel_mapping.csv')
fuel_mapping.set_index('in_stack', drop = True, inplace=True)
fuel_dict = fuel_mapping.to_dict()
Fuel_Data = pd.read_csv('FUEL_PRICES_DATA.csv')

parameters = {
    "ARIMA":
        {"p": [1,2,3],
        "q": [4,5,6],
        "d": [0]
       },

    "ExponentialSmoothing":
        {"trend": [ModelMode.NONE, ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE]},

    "T":
        {
          "theta":[ 0.4, 0.5, 0.6],
          "model_mode": [ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE],
          "trend_mode": [TrendMode.LINEAR, TrendMode.EXPONENTIAL]
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

def Best_model_V2(model_list, train, metric = 'MAPE', exog = None):
  results = {}
  Max_Dev = float("Inf")
  for m in model_list:
    r = m.historical_forecasts(train, start = START_POINT, forecast_horizon = 14, verbose = False,  future_covariates = exog)
    met =  metrics.metrics.mape(r, train)
    print(metric,' :', met)
    if met < Max_Dev:
      Max_Dev = met
      results['Output'] = r
      results['Model'] = m
  print('Best Model: ',results['Model'])
  return results

def To_Date_Data(Data, var_list):
  Today = datetime.date.today() 
  Forecast_Date = Today + timedelta(days = 13)
  End_Date =  pd.to_datetime(Data['Date'].iloc[-1])
  print('Today: ', Today)
  print('Forecast Date: ', Forecast_Date)
  print('End Date: ', End_Date)

  Forecast_Date =  pd.to_datetime(Forecast_Date)
  FORECAST_PERIOD = max((Forecast_Date - End_Date).days,0)

  for var in var_list:
    Fuel = fuel_dict['in_fuel_price'][var]
    df = Fuel_Data[['Date', Fuel]]
    df['Date'] = pd.DatetimeIndex(df['Date'])
    Start_Date = pd.datetime(2021,5,1)
    df = df.loc[(df['Date'] <= End_Date) & (df['Date'] >= Start_Date)]
    temp['Date'] = pd.DatetimeIndex(temp['Date'])    
    series = TimeSeries.from_dataframe(df, time_col = 'Date', freq = 'd', fill_missing_dates = True)
    series.plot(label = Fuel)
    temp_serie = TimeSeries.from_dataframe(temp, time_col = 'Date', freq = 'd', fill_missing_dates = True)
    models_list = []
    Tet = FourTheta.gridsearch(parameters = parameters["T"], series = series, forecast_horizon = FORECAST_PERIOD, metric = metrics.metrics.mape, n_jobs = -1 ,  future_covariates=temp_serie)
    models_list.append(Tet[0])
    Arima_model = ARIMA.gridsearch(parameters = parameters["ARIMA"], forecast_horizon = FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
    models_list.append(Arima_model[0])
    Expon = ExponentialSmoothing.gridsearch(parameters = parameters["ExponentialSmoothing"], forecast_horizon= FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
    models_list.append(Expon[0])
    a = Best_model_V2(models_list, series, metric = 'MAPE', exog = temp_serie)
    a['Output'].plot(label = str(a["Model"]))
    winsound.Beep(440, 100)
    plt.ylim(0,10)
    plt.savefig('model.pdf')
    plt.show()

  
To_Date_Data(Fuel_Data, Fuel_list)

'''
df = Fuel_Data[['Date', Fuel]]
Start_Date = pd.datetime(2021,5,1)

Forecast_Date = Today + datetime.timedelta(days = FORECAST_PERIOD)

df['Date'] = pd.DatetimeIndex(df['Date'])
#df = df.loc[(df['Date'] <= End_Date) & (df['Date'] >= Start_Date)]
df = df.loc[df['Date'] >= Start_Date]
temp['Date'] = pd.DatetimeIndex(temp['Date'])

series = TimeSeries.from_dataframe(df, time_col = 'Date', freq = 'd', fill_missing_dates = True)
series.plot(label = Fuel)
temp_serie = TimeSeries.from_dataframe(temp, time_col = 'Date', freq = 'd', fill_missing_dates = True)

models_list = []
Tet = FourTheta.gridsearch(parameters = parameters["T"], series = series, forecast_horizon = FORECAST_PERIOD, metric = metrics.metrics.mape, n_jobs = -1 ,  future_covariates=temp_serie)
models_list.append(Tet[0])
Arima_model = ARIMA.gridsearch(parameters = parameters["ARIMA"], forecast_horizon = FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
models_list.append(Arima_model[0])
Expon = ExponentialSmoothing.gridsearch(parameters = parameters["ExponentialSmoothing"], forecast_horizon= FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
models_list.append(Expon[0])
a = Best_model_V2(models_list, series, metric = 'MAPE', exog = temp_serie)
a['Output'].plot(label = str(a["Model"]))
winsound.Beep(440, 100)
plt.ylim(0,7)
plt.savefig('model.pdf')
plt.show()



#Auto_Arima = AutoARIMA()
#Auto_Arima.fit(series = series,  future_covariates = temp_serie)
#Varima = VARIMA.gridsearch(parameters = parameters["VARIMA"], forecast_horizon = FORECAST_PERIOD, series = series)
#Kalman = KalmanForecaster.gridsearch(series = series,parameters = parameters["KALMAN"],  forecast_horizon = FORECAST_PERIOD,  future_covariates=temp_serie)

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

h = Varima[0].historical_forecasts(series, start = 0.8, forecast_horizon= FORECAST_PERIOD, verbose = False,  future_covariates=temp_serie)
mape = metrics.metrics.mape(h, series)
h.plot(label = 'Varima: '+ str(mape))

'''
