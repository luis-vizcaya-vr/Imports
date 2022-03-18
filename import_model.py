from sklearn import *
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime
from datetime import date, timedelta

def Get_Predict(Data, Y_Name, Forecast_Period):
    train_len = int(len(Data))
    imp = Data[[Y_Name]]
    Data.drop([Y_Name], axis=1, inplace=True)
    loads_train = Data[:train_len - Forecast_Period]
    imp_train = imp[:train_len - Forecast_Period]
    loads_test = Data[train_len - Forecast_Period:]
    #print('loads_train', loads_train)
    
    regr = linear_model.LinearRegression()
    regr.fit(loads_train, imp_train)
    pred = regr.predict(loads_test)
    return pred

DAYS_TO_FORECAST = 15
FORECAST_PERIOD = DAYS_TO_FORECAST * 24                                                                   #total hours to forecast
DATA_RAW = pd.read_csv('Import_Data_Raw_v5.csv')                                            #input file
Today = datetime.date.today()
Forecast_Date = Today + timedelta(days = DAYS_TO_FORECAST )


DATA_RAW.dropna(inplace=True)                                                               #drop nan values
#Y_Name = 'PJM: Tie-Flows MISO imports 5Min(7732)-MW'                                                 #dependent variable
#Columns_Drop = 'PJM: InterTieFlow-NYIS(6096)-MW'                                #drop the other zonal import

Y_List = ['PJM: Tie-Flows Exports 5Min(6553)-MW']

DATA_RAW.set_index(['Date'],  inplace=True, drop = True)

DATA_RAW['TimeLabel'] = DATA_RAW['DayOfWeek'].apply(str) + DATA_RAW['Hour'].apply(str)      #label created to map weekday-hour average for seasonal behavior
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
AVGS = DATA_RAW[:-FORECAST_PERIOD].groupby(['DayOfWeek', 'Hour'], as_index = False).mean()  #calculating average for each hour 
AVGS.reset_index(inplace=True, drop = True)                                                 #reset the index
AVGS['TimeLabel'] = AVGS['DayOfWeek'].apply(str) + AVGS['Hour'].apply(str)                  #label created to map weekday-hour average for removing seasonality
STDVS = DATA_RAW[:-FORECAST_PERIOD].groupby(['DayOfWeek', 'Hour'], as_index = False).std()  #calculating std for each hour
STDVS.reset_index(inplace=True, drop = True)                
STDVS['TimeLabel'] = STDVS['DayOfWeek'].apply(str) + STDVS['Hour'].apply(str)               #label created to map weekday-hour average for removing seasonality

for c in DATA_RAW.columns[2:-1]:
    AVG2[c] = DATA_RAW['TimeLabel'].map(AVGS.set_index('TimeLabel')[c])
    STD2[c] = DATA_RAW['TimeLabel'].map(STDVS.set_index('TimeLabel')[c])

DATA_RAW.drop(['TimeLabel','DayOfWeek','Hour'], axis=1, inplace=True)
loads = DATA_RAW.copy()                                     #taking the data only
loads = (loads - AVG2)/STD2                                     #normalizing the data
loads.to_csv('Normalized_data.csv')
#print(loads.head())
#print('Today', Today)
#print('Hoy:',loads.loc[:,Today])


#loads = loads[(np.abs(stats.zscore(loads)) < 2).all(axis=1)]    #removing outliers greater than z= 2

#loads.reset_index(inplace = True, drop = True)                  #reset the in

#print('Y_list',Y_List)
pd_output = pd.DataFrame(index = DATA_RAW.index[-FORECAST_PERIOD:])
#print(pd_output)
for y in Y_List:
#    print('Forecasting: ', y)
    Aux_Load = loads.copy()
    Aux_List = Y_List.copy()
    imp = Aux_Load[[y]].copy()
    #print('imp: ',imp)
    Aux_List.remove(y)
    #print('Aux_list:', Aux_List)
    Aux_Load.drop(Aux_List, axis=1, inplace=True)                  
    imp_pred = Get_Predict(Aux_Load, y, FORECAST_PERIOD)
    train_len = int(len(Aux_Load))
    imp_test = imp[train_len-FORECAST_PERIOD:].values

    AVG2_TEST = AVG2[-FORECAST_PERIOD:][y]
    AVG2_TEST = pd.DataFrame(AVG2_TEST)
    AVG2_TEST.reset_index(drop = True, inplace = True)
    STD2_TEST = STD2[-FORECAST_PERIOD:][y]
    STD2_TEST = pd.DataFrame(STD2_TEST)
    STD2_TEST.reset_index(drop = True, inplace = True)
    imp_pred = imp_pred * STD2_TEST + AVG2_TEST
    imp_test = imp_test * STD2_TEST + AVG2_TEST
    
    #DATA_RAW.loc[-FORECAST_PERIOD:, y] = imp_pred.values
    pd_output.loc[:,y] =imp_pred.values
    #DATA_RAW.iloc[-FORECAST_PERIOD:,11] = imp_pred

    
    imp_pred = pd.DataFrame(imp_pred, columns={y})
    imp_pred.reset_index(inplace = True, drop = True)
    imp_test = pd.DataFrame(imp_test, columns={y})
    imp_test.reset_index(inplace = True, drop = True)
    print(mean_absolute_percentage_error(imp_test,imp_pred))
 #   print('imp_pred', imp_pred)
    
    #pd_output.loc[:,y] = imp_pred.values()
    #DATA_RAW[-FORECAST_PERIOD:][y] = imp_pred.values()
    
    plt.plot(imp_pred, label = 'Prediction')
    plt.plot(imp_test, label = 'Real Data')
    plt.legend()
    plt.show()
    
print(DATA_RAW.tail())
print(pd_output.head())
    
pd_output.to_csv('output.csv')
# train_len = int(len(loads))
# loads.drop([Y_Name], axis=1, inplace=True)
# regr = linear_model.LinearRegression()
# loads_train = loads[:train_len-FORECAST_PERIOD]
# loads_test = loads[train_len-FORECAST_PERIOD:]
# imp_train = imp[:train_len-FORECAST_PERIOD]
# regr.fit(loads_train, imp_train)
# imp_pred = regr.predict(loads_test)

