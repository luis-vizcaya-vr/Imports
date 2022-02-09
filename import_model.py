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


DATA_RAW = pd.read_csv('Import_Data_v5.csv')
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

loads = loads[(np.abs(stats.zscore(loads)) < 1.5).all(axis=1)]

loads.dropna(inplace = True)
loads = loads.diff()[1:]
imp = loads[['IMPORT-PJM']]

#elim = [4,5,9,11]
#loads.drop(loads.columns[elim],axis = 1, inplace = True)
#loads.drop(['IMPORT-PJM','LMP-WEST-PJM','LMP-WESTERN-PJM','LMP-MINNESOTA-MISO','LMP-ILLINOIS-PJM','LMP-MICHIGAN-PJM','LMP-COMED-PJM','LMP-AEP-PJM'], axis=1, inplace=True)
loads.drop(['IMPORT-PJM'], axis=1, inplace=True)

regr = linear_model.LinearRegression()
train_len = int(len(loads)*0.8)
print('train_len:', train_len)
loads_train = loads[:train_len]
print('len(load_train): ', len(loads_train))
loads_test = loads[train_len:]
imp_train = imp[:train_len]
imp_test = imp[train_len:]

#loads_train, loads_test, imp_train, imp_test = train_test_split(loads, imp, test_size=0.2, random_state=0)
#print('load_train: ',loads_test)
#print('imp_train:',imp_test)



regr.fit(loads_train, imp_train)
imp_pred = regr.predict(loads_test)
print('R2: before SARIMA',r2_score(imp_test,imp_pred))

pred_train= regr.predict(loads_train)   # predicted values for training data

residual = imp_train - pred_train       # residual in training data


plt.plot(imp)
#plt.legend()

acf_vals = acf(residual)
num_lags = 20
#plt.bar(range(num_lags), acf_vals[:num_lags])
plt.show()


