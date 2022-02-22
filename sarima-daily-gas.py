from sklearn import linear_model
from datetime import datetime,date
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from forecast import * 

AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
DATA_RAW = pd.read_csv('NATGAS-DATA.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW.reset_index()
Y = DATA_RAW['NatGas - Transco Z6 Non-NY']
Data_len = len(Y)
print('Data_len:',Data_len)
#plt.plot(DATA_RAW['IMPORT-PJM'])
#plt.show()
print('Y: ', Y)
FORECAST_PERIOD = 14
ps = range(1 ,2)
qs = range(1 ,2)
Ps = range(1 ,2)
Qs = range(1 ,2)
Ss = range(7, 15, 7)
starts = range(0,35,7)


R = Hyper_Param_Sarima(Y, ps,qs,Ps,Qs,Ss,starts,FORECAST_PERIOD)
Best_Par = R['Best_Par']
Y_train = Y[Best_Par['start']:Data_len-FORECAST_PERIOD]
Y_test = Y[Data_len - FORECAST_PERIOD:]
Forecast = Evaluate_Sarima(Y_train, Y_test, Best_Par['p'], 1, Best_Par['q'], Best_Par['P'], 0, Best_Par['Q'], Best_Par['S'], FORECAST_PERIOD)
Predictions = Y.copy()
Y[Data_len - FORECAST_PERIOD:] = Forecast['Forecast']

#Predictions = list(Forecast['Forecast'])
#Y_test = list(Y_test)
#residuals = [float(a_i) - float(b_i) for a_i, b_i in zip(Imp_test, Predictions)]  

print('MAE', Forecast['MAE'])
print('MAX ERROR', Forecast['MAX_ERROR'])
print('%MAE', Forecast['MAE']/np.average(Y_test))
#plt.axis([0,14,0,6])
plt.plot(Y, label = 'Real')
plt.plot(Predictions, label = 'Prediction')
plt.show()



