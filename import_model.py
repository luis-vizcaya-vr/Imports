from sklearn import *
from sklearn.metrics import max_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error
from forecast import *

FORECAST_PERIOD = 336
DATA_RAW = pd.read_csv('Import_Data_v6.csv')
DATA_RAW.dropna(inplace=True)
Y_Name = 'IMPORT-NY'
Columns_Drop = ['IMPORT-PJM']

DATA_RAW['TimeLabel'] = DATA_RAW['DayOfWeek'].apply(str) + DATA_RAW['Hour'].apply(str)      #label created to map weekday-hour average
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
AVGS = DATA_RAW[:-FORECAST_PERIOD].groupby(['DayOfWeek', 'Hour'], as_index = False).mean() #calculating average for each hour 
AVGS.reset_index(inplace=True, drop = True)                                                      #reset the index
AVGS['TimeLabel'] = AVGS['DayOfWeek'].apply(str) + AVGS['Hour'].apply(str)
STDVS = DATA_RAW[:-FORECAST_PERIOD].groupby(['DayOfWeek', 'Hour'], as_index = False).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True, drop = True)                
STDVS['TimeLabel'] = STDVS['DayOfWeek'].apply(str) + STDVS['Hour'].apply(str)

for c in DATA_RAW.columns[3:-1]:
    AVG2[c] = DATA_RAW['TimeLabel'].map(AVGS.set_index('TimeLabel')[c])
    STD2[c] = DATA_RAW['TimeLabel'].map(STDVS.set_index('TimeLabel')[c])

DATA_RAW.drop(['TimeLabel'], axis=1, inplace=True)

loads = DATA_RAW.iloc[:,3:]                                     #taking the data only
loads = (loads - AVG2)/STD2                                     #
loads = loads[(np.abs(stats.zscore(loads)) < 2).all(axis=1)]    #removing outliers
loads = loads.diff()[1:]                                        #applying first difference for removing time trend
loads.reset_index(inplace = True)                               #reset the index
        
MAX_R2 =  0
MAX_MAE = 5000000
MAX_MAX_ERROR = 5000000
best_col =-1
best_cut = 0
col_aux = -1
loads.drop(Columns_Drop, axis=1, inplace=True)
aux_loads = loads.copy()
loads_orig = loads.copy()
    
#for col in range(1,len(loads.columns)-1):
#    print('running col:', col)
cut = 0
while (cut <1000):
    loads = aux_loads[cut:len(aux_loads)].copy()
    train_len = int(len(loads))
    #print('train_len : ', train_len)
    imp = loads[[Y_Name]]
    print('imp: ',imp)
    loads.drop([Y_Name], axis=1, inplace=True)
    regr = linear_model.LinearRegression()
    loads_train = loads[:train_len-FORECAST_PERIOD]
    loads_test = loads[train_len-FORECAST_PERIOD:]
    imp_train = imp[:train_len-FORECAST_PERIOD]
    imp_test = imp[train_len-FORECAST_PERIOD:]
    regr.fit(loads_train, imp_train)
    imp_pred = regr.predict(loads_test)
    MAE = mean_absolute_error(imp_test, imp_pred)
    MAX_ERROR = max_error(imp_test, imp_pred)
    
    if MAX_MAE > MAE:
        print('col_aux:',col_aux)
        MAX_MAE = MAE
        best_cut = cut
        best_col = col_aux
            
    cut += 168
#    col_aux = col
#    aux_loads = loads_orig.drop(loads_orig.columns[col_aux],axis = 1).copy()

#if best_col >-1:
#    loads = loads_orig.drop(loads_orig.columns[best_col],axis = 1)
#    print('best col to drop: ', loads.columns[best_col])

AVG2_TEST = AVG2[-FORECAST_PERIOD:][Y_Name]
AVG2_TEST = pd.DataFrame(AVG2_TEST)
AVG2_TEST.reset_index(drop = True, inplace = True)
STD2_TEST = STD2[-FORECAST_PERIOD:][Y_Name]
STD2_TEST = pd.DataFrame(STD2_TEST)
STD2_TEST.reset_index(drop = True, inplace = True)

print('AVG2_Test: ',AVG2_TEST)
print('best col N: ', best_col)
print('best cut:',best_cut)
print('Max R2:',MAX_R2)
print('MAE:', MAX_MAE)
print('STD2_TEST:',STD2_TEST)
print('imp_pred:',imp_pred)
print('imp_test:',imp_test)

imp_pred = pd.DataFrame(imp_pred, columns={Y_Name})
imp_pred.reset_index(inplace = True, drop = True)

imp_test = pd.DataFrame(imp_test, columns={Y_Name})
imp_test.reset_index(inplace = True, drop = True)

imp_pred = imp_pred * STD2_TEST + AVG2_TEST
imp_test = imp_test * STD2_TEST + AVG2_TEST
MAE = mean_absolute_error(imp_test, imp_pred)
PERC_MAE = MAE/imp_test.mean()
print('MAE:', MAE)
print('% MAE:', PERC_MAE)

plt.plot(imp_pred, label = 'Prediction')
plt.plot(imp_test, label = 'Real Data')
plt.legend()
plt.show()