from re import M
#from tkinter.tix import Tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


DATA_RAW = pd.read_csv('Import_Data_v4.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW = DATA_RAW.iloc[len(DATA_RAW)-1000:-1,:]

loads = DATA_RAW[['DEMAND-MISO','DEMAND-PJM','WIND-MISO','WIND-PJM','IMPORT-PJM']]

AVGS = DATA_RAW.groupby(['DayOfWeek', 'Hour'], as_index = False).mean()
AVGS.reset_index() #reset the index
#print(AVGS.loc[ (AVGS.DayOfWeek=='Monday') & (AVGS.Hour ==0)])


AVGS1 = DATA_RAW.index.map(lambda d: AVGS.loc[ (AVGS.DayOfWeek==d.DayOfWeek) & (AVGS.Hour==d.Hour)])

print( AVGS1)


#AVGS.reset_index(inplace=True)

STDVS = DATA_RAW.groupby(['DayOfWeek', 'Hour']).std()
STDVS.reset_index(inplace=True)

DEVS = pd.DataFrame()
#print(AVGS)


DATA = pd.DataFrame()
for r in loads.index:
    for c in loads.columns:

        DIA = DATA_RAW['DayOfWeek'][r]
        HORA = DATA_RAW['Hour'][r]
        a = (AVGS.loc[:,'DayOfWeek']==DIA) & (AVGS.loc[:,'Hour']==HORA)
        PROM = AVGS.loc[a]
        a = (STDVS.loc[:,'DayOfWeek']==DIA) & (STDVS.loc[:,'Hour']==HORA)
        SD = STDVS.loc[a]
        
        PROM = float(PROM[c])
        SD = float(SD[c])
        
        
        #loads.loc[r,c] = (loads.loc[r,c]-PROM)/SD

imp = loads[['IMPORT-PJM']]
loads.drop('IMPORT-PJM', axis=1, inplace=True)

    
regr = linear_model.LinearRegression()
loads_train, loads_test, imp_train, imp_test = train_test_split(loads, imp, test_size=0.2, random_state=0)


regr.fit(loads_train, imp_train)
imp_pred = regr.predict(loads_test)



print('R2: ',r2_score(imp_test,imp_pred))


