from pickle import TRUE
from re import M
#from tkinter.tix import Tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf


DATA_RAW = pd.read_csv('Import_Data_v5.csv')
DATA_RAW.dropna(inplace=True)
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
#plt.plot(DATA_RAW[['IMPORT-PJM']])
#plt.show()


AVGS = DATA_RAW.groupby(['DayOfWeek', 'Hour'], as_index = False).mean() #calculating average for each hour 
AVGS.reset_index()                                                      #reset the index
STDVS = DATA_RAW.groupby(['DayOfWeek', 'Hour']).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True)                                         #reset the index

#normalizing and removing seasonality
#for c in DATA_RAW.columns[3:]:
#    AVG2[c] = [ float(AVGS[(AVGS.DayOfWeek==x) & (AVGS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
#    STD2[c] = [ float(STDVS[(STDVS.DayOfWeek==x) & (STDVS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
#    DATA_RAW[c] = (DATA_RAW[c]-AVG2[c])/STD2[c]

DATA_RAW = DATA_RAW[:-1] #droping the last row due to noise


loads = DATA_RAW.iloc[:,3:]                                     #taking the data only
loads = loads[(np.abs(stats.zscore(loads)) < 2).all(axis=1)]    #removing outliers
loads.dropna(inplace = True)                                    #removing nan values
loads.reset_index(inplace = True)                               #reset the index
#loads = loads.diff()[1:]                                        #applying first difference for removing trend
imp = loads[['IMPORT-PJM']]
loads.drop(['IMPORT-PJM'], axis=1, inplace=True)       
train_len = int(len(imp))
print('train_len: ',train_len)         
imp_train = imp[:train_len-168]
imp_test = imp[train_len-168:]

acf_vals = acf(imp_train)
num_lags = 30
#plt.bar(range(num_lags), acf_vals[:num_lags])

pacf_vals = pacf(imp_train)
#plt.bar(range(num_lags), pacf_vals[:num_lags])

my_order = (0,1,0)
my_seasonal_order = (1, 0, 1, 12)
# define model
model = SARIMAX(imp_train, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()
print(model_fit.summary())
predictions = model_fit.forecast(len(imp_test))
predictions = pd.Series(predictions)
predictions.reset_index(inplace = True, drop=True)
imp_test.reset_index(inplace = True, drop=True)

#predictions = list(predictions)
imp_test = list(imp_test['IMPORT-PJM'])

#residuals = [float(a_i) - float(b_i) for a_i, b_i in zip(imp_test, predictions)]  


#print('pred', predictions)
print('imp_test', imp_test)


plt.plot(imp_test)
plt.plot(predictions)
plt.show()
        