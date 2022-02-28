
from sklearn import linear_model
from datetime import datetime,date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error

DATA_RAW = pd.read_csv('Import_Data_daily.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW.reset_index()
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
AVGS = DATA_RAW.groupby(['DayOfWeek']).mean()                   #calculating average for each hour 
AVGS.reset_index(inplace=True)                                  #reset the index
STDVS = DATA_RAW.groupby(['DayOfWeek']).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True)   
imp = DATA_RAW['IMPORT-PJM']
imp = (imp -imp.mean())/imp.std()
print(imp)



MIN_MAE = 50000
train_len = int(len(imp))
    

for start in range(0,35,7):
    imp_train = imp[start:train_len-14]
    imp_test = imp[train_len-14:]
    

    for p in range(1,5):
        for q in range(1,5):
            for P in range(1,5):
                for Q in range(1,5):
                    #for S in range(7,15,7):
                    print('PARAMS:',[p,q,P,Q,7,start])
                    my_order = (p,1,q)
                    my_seasonal_order = (P, 0, Q, 7)
                    model = SARIMAX(imp_train, order=my_order, seasonal_order=my_seasonal_order)
                    model_fit = model.fit()
                    predictions = model_fit.forecast(len(imp_test))
                    predictions = pd.Series(predictions)
                    predictions.reset_index(inplace = True, drop=True)
                    imp_test.reset_index(inplace = True, drop=True)
                    MAE = mean_absolute_error(imp_test, predictions)
                    if MAE < MIN_MAE:
                        MIN_MAE = MAE
                        Best_Par = [p,q,P,Q,7,start] 


imp_train = imp[start:train_len-14]
imp_test = imp[train_len-14:]
my_order = (Best_Par[0],1,Best_Par[1])
my_seasonal_order = (Best_Par[2], 0, Best_Par[3], 7)
model = SARIMAX(imp_train, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()
predictions = model_fit.forecast(len(imp_test))
predictions = pd.Series(predictions)
predictions.reset_index(inplace = True, drop=True)

#R2 = r2_score(imp_test, predictions)

#predictions = list(predictions)
#imp_test = list(imp_test['IMPORT-PJM'])

residuals = [float(a_i) - float(b_i) for a_i, b_i in zip(imp_test, predictions)]  


print('MAE', MIN_MAE)
print('Best Parameters [p,q,P,Q,S]', Best_Par)
#print('R2', R2)

#print('imp_test', imp_test)


plt.plot(imp_test)
plt.plot(predictions)
plt.show()
# 



