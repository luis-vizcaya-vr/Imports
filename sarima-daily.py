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
DATA_RAW = pd.read_csv('Import_Data_daily.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW.reset_index()
Imp = DATA_RAW['IMPORT-PJM']
Data_len = len(Imp)
print('Data_len:',Data_len)
#plt.plot(DATA_RAW['IMPORT-PJM'])
#plt.show()
print('Imp', Imp)
FORECAST_PERIOD = 14
ps = range(1 ,4)
qs = range(1 ,4)
Ps = range(1 ,4)
Qs = range(1 ,4)
Ss = range(7, 15, 7)
starts = range(0,28,7)


print(get_imports2('PJM', backtest_input_dict))

R = Hyper_Param_Sarima(Imp, ps,qs,Ps,Qs,Ss,starts,FORECAST_PERIOD)
Best_Par = R['Best_Par']

Imp_train = Imp[Best_Par['start']:Data_len-FORECAST_PERIOD]
Imp_test = Imp[Data_len - FORECAST_PERIOD:]
Forecast = Evaluate_Sarima(Imp_train, Imp_test, Best_Par['p'], 1, Best_Par['q'], Best_Par['P'], 0, Best_Par['Q'], Best_Par['S'], FORECAST_PERIOD)
Predictions = list(Forecast['Forecast'])
Imp_test = list(Imp_test)
residuals = [float(a_i) - float(b_i) for a_i, b_i in zip(Imp_test, Predictions)]  

print('MAE', Forecast['MAE'])
print('MAX ERROR', Forecast['MAX_ERROR'])
print('%MAE', Forecast['MAE']/np.average(Imp_test))

plt.plot(Imp_test)
plt.plot(Predictions)
plt.show()



