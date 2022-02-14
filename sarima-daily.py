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
#plt.plot(DATA_RAW['IMPORT-PJM'])
#plt.show()

Imp = DATA_RAW['IMPORT-PJM']
print('Imp', Imp)
FORECAST_PERIOD = 7
ps = range(1 ,3)
qs = range(1 ,3)
Ps = range(1 ,3)
Qs = range(1 ,3)
Ss = range(7, 15, 7)
starts = range(0,28,7)

R = Hyper_Param_Sarima(Imp, ps,qs,Ps,Qs,Ss,starts,FORECAST_PERIOD)
Best_Par = R['Best_Par']

print('Best Parameters [p,q,P,Q,S]', Best_Par)
imp_train = imp[Best_Par[5]:train_len-FORECAST_PERIOD]
imp_test = imp[train_len-FORECAST_PERIOD:]
my_order = (Best_Par[0],1,Best_Par[1])
my_seasonal_order = (Best_Par[2], 0, Best_Par[3], Best_Par[4])
model = SARIMAX(imp_train, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit(disp = False)
predictions = model_fit.forecast(FORECAST_PERIOD)
predictions = pd.Series(predictions)
predictions.reset_index(inplace = True, drop=True)
MAE = mean_absolute_error(imp_test, predictions)
predictions = list(predictions)
imp_test = list(imp_test)

residuals = [float(a_i) - float(b_i) for a_i, b_i in zip(imp_test, predictions)]  

print('MAE', MAE)
print('MAX ERROR', MIN_MAX_ERROR)
print('%MAE', MAE/np.average(imp_test))

plt.plot(imp_test)
plt.plot(predictions)
plt.show()
 """



