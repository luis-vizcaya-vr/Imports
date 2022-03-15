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

parameters = {
    "ARIMA":
        {"p": [1],
        "q": [5],
        "d": [0]
       },

    "ExponentialSmoothing":
        {"trend": [ModelMode.NONE, ModelMode.MULTIPLICATIVE, ModelMode.ADDITIVE]},

    "T":
        {
          "theta":[0.3, 0.4, 0.5, 0.6, 1],
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

START_POINT = 0.92
FORECAST_PERIOD = 14
Today = datetime.date.today()
Forecast_Date = Today + timedelta(days = FORECAST_PERIOD)
Fuel_list = list(pd.read_csv('FUEL_SIDS.csv')['FUEL'])
#print('Fuel_list: ', Fuel_list)
Fuel_list = Fuel_list[0:2]
temp = pd.read_csv('NATGAS-DATA_V3.csv', usecols= [0,11])
fuel_mapping = pd.read_csv('fuel_mapping.csv')
fuel_mapping.set_index('in_stack', drop = True, inplace=True)
fuel_dict = fuel_mapping.to_dict()
Fuel_Data = pd.read_csv('FUEL_PRICES_DATA.csv')
Fuel_Data['Date'] = pd.DatetimeIndex(Fuel_Data['Date'])

def Best_model_V2(model_list, train, metric = 'MAPE', exog = None, period =14):
  results = {}
  Max_Dev = float("Inf")
  i = 0
  for m in model_list:
    r = m.historical_forecasts(train, start = START_POINT, forecast_horizon = period, verbose = False, past_covariates = exog, future_covariates = exog)
    met =  metrics.metrics.mse(r, train)
    results['MAPE'] = metrics.metrics.mape(r, train)
    if met < Max_Dev:
      Max_Dev = met
      results['Output'] = r
      results['Model'] = m
      results['metric'] = met
    i+=1
  print('Best Model: ',results['Model'])
  return results

def To_Date_Data(Data, var_list, period =14, mode = 'FORECAST'):
  Num_Var = len(var_list)
  fig, axes = plt.subplots(nrows= Num_Var, ncols=1)
  Today = datetime.date.today() 
  Forecast_Date = Today + timedelta(days = period)
  Start_Date = pd.datetime(2021,5,1)
  Forecast_Date =  pd.to_datetime(Forecast_Date)
  test = Data['Date'].dropna()
  Last_Index = test.index[-1]
  Last_Date = Data.loc[Last_Index,'Date']
  Last_Date = Last_Date 
  d_range = pd.date_range(start = Last_Date, end = Forecast_Date)
  i = 0    
  for d in d_range:
    Data.loc[Last_Index + i , 'Date'] = d
    i+= 1
  subp = 1
  for col in Data.columns[1:]:
    test = Data[col].dropna()
    Last_Index = test.index[-1]
    Last_Date = Data.loc[Last_Index,'Date']
    Last_Date = Last_Date 
    Last_Value = Data.loc[Last_Index, col]
    FORECAST_PERIOD = max((Forecast_Date - Last_Date).days,0)
    d_range = pd.date_range(start = Last_Date + timedelta(days = 1), end = Forecast_Date)
    
    if col not in var_list:
      i = 1
      for d in d_range:
        Data.loc[Last_Index + i , col] = Last_Value 
        i+= 1
    
    else:  
      df = Data[['Date', col]]
      df['Date'] = pd.DatetimeIndex(df['Date'])
      df = df.loc[(df['Date'] <= Last_Date) & (df['Date'] >= Start_Date)]
      temp['Date'] = pd.DatetimeIndex(temp['Date'])    
      series = TimeSeries.from_dataframe(df, time_col = 'Date', freq = 'd', fill_missing_dates = True)
      temp_serie = TimeSeries.from_dataframe(temp, time_col = 'Date', freq = 'd', fill_missing_dates = True)
      models_list = []
      Arima_model = ARIMA.gridsearch(parameters = parameters["ARIMA"], forecast_horizon = FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
      models_list.append(Arima_model[0])
      Best_Model = Best_model_V2(models_list, series, metric = 'MAPE', exog = temp_serie, period = FORECAST_PERIOD)  
      Best_Model['Model'].fit(series, future_covariates=temp_serie)
      if mode == 'BACKTEST':
        f = Best_Model['Model'].historical_forecasts(series, start = START_POINT, forecast_horizon = 14, verbose = False,  future_covariates = temp_serie)
      else:
        f = Best_Model['Model'].predict(FORECAST_PERIOD, future_covariates=temp_serie)
      
      f.to_csv('time_series.csv')      
      df_forecast = f.pd_dataframe()
      
      #axs[subp] = f.plot()
      #plt.plot(f.values())
      plt.subplot(Num_Var,1,subp)
      plt.plot(series.pd_dataframe()) 
      plt.plot(df_forecast, label = col) 
      #plt.plot(f) 
      #ax[subp,0].plot(series) 
      #series.plot(ax = axes[subp])
      #axs[subp] = f.plot()
      i = 1
      for d in d_range:
        Data.loc[Last_Index + i , col] = df_forecast.iloc[i-1, 0]
        i+= 1

      subp += 1
  winsound.Beep(440, 150)
  plt.savefig('Gas_Price.pdf')
  plt.show()
  Data.to_csv('Fuel_Price_Forecast.csv')

To_Date_Data(Fuel_Data, Fuel_list, 'BACKTEST')


'''
#Expon = ExponentialSmoothing.gridsearch(parameters = parameters["ExponentialSmoothing"], forecast_horizon= FORECAST_PERIOD, series = series,  verbose = False, n_jobs = -1,  future_covariates=temp_serie)
#models_list.append(Expon[0])
#Tet = FourTheta.gridsearch(parameters = parameters["T"], series = series, forecast_horizon = FORECAST_PERIOD, metric = metrics.metrics.mape, n_jobs = -1 ,  future_covariates=temp_serie)
#models_list.append(Tet[0])
      
Last_Index = Fuel_Data.index[-1]
Last_Date = Fuel_Data.iloc[Last_Index, 0]
Today = datetime.date.today()
Forecast_Date = Today + timedelta(days = FORECAST_PERIOD)
Forecast_Date =  pd.to_datetime(Forecast_Date)
print('Last Index: ', Last_Index)
print('Last Date: ', Last_Date)
print('Forecast Date: ', Forecast_Date)
  
d_range = pd.date_range(start = Last_Date, end = Forecast_Date , freq='D')
print('d_range', d_range)
    
i = 1
for d in d_range:
  Fuel_Data.loc[Last_Index + i , 'Date'] = d
  i+= 1
print(Fuel_Data.tail())

Fuel_Data.to_csv('Output_file.csv')
    
    


#print('Fuel List:', Fuel_list)
#To_Date_Data(Fuel_Data, Fuel_list, FORECAST_PERIOD)


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
