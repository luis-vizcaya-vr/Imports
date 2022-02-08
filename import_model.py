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
#print(DATA_RAW.columns[1])
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
#DATA_RAW = DATA_RAW.iloc[len(DATA_RAW)-1000:-1,:]
#print(DATA_RAW.index)
#A = DATA_RAW.apply(lambda x: AVGS[(AVGS.DayOfWeek==x.DayOfWeek) & (AVGS.Hour==1)]['DEMAND-MISO'],axis = 1)
#print(AVGS.loc[(AVGS.DayOfWeek=='Monday') & (AVGS.Hour==0)])
#DATA_RAW['PRUEBA'] = DATA_RAW['DayOfWeek', 'Hour'].map(AVGS.set_index(['DayOfWeek', 'Hour'])['DEMAND-MISO'])
#DATA_RAW.set_index(['DayOfWeek', 'Hour'])

AVGS = DATA_RAW.groupby(['DayOfWeek', 'Hour'], as_index = False).mean()
AVGS.reset_index() #reset the index
STDVS = DATA_RAW.groupby(['DayOfWeek', 'Hour']).std()
STDVS.reset_index(inplace=True)

for c in DATA_RAW.columns[3:]:
    AVG2[c] = [ float(AVGS[(AVGS.DayOfWeek==x) & (AVGS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
    STD2[c] = [ float(STDVS[(STDVS.DayOfWeek==x) & (STDVS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
    DATA_RAW[c] = (DATA_RAW[c]-AVG2[c])/STD2[c]

DATA_RAW = DATA_RAW[:-1]

loads = DATA_RAW.iloc[:,3:]
print(loads)
imp = loads[['IMPORT-PJM']]
loads.drop('IMPORT-PJM', axis=1, inplace=True)
  
regr = linear_model.LinearRegression()
loads_train, loads_test, imp_train, imp_test = train_test_split(loads, imp, test_size=0.2, random_state=0)


regr.fit(loads_train, imp_train)
imp_pred = regr.predict(loads_test)



print('R2: ',r2_score(imp_test,imp_pred))

