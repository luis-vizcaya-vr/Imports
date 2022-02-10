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
AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
#plt.plot(DATA_RAW[['IMPORT-PJM']])
plt.show()


AVGS = DATA_RAW.groupby(['DayOfWeek', 'Hour'], as_index = False).mean() #calculating average for each hour 
AVGS.reset_index()                                                      #reset the index
STDVS = DATA_RAW.groupby(['DayOfWeek', 'Hour']).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True)                                         #reset the index

#normalizing and removing seasonality
for c in DATA_RAW.columns[3:]:
    AVG2[c] = [ float(AVGS[(AVGS.DayOfWeek==x) & (AVGS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
    STD2[c] = [ float(STDVS[(STDVS.DayOfWeek==x) & (STDVS.Hour==y)][c]) for x,y in zip(DATA_RAW['DayOfWeek'],DATA_RAW['Hour'])]
    DATA_RAW[c] = (DATA_RAW[c]-AVG2[c])/STD2[c]

DATA_RAW = DATA_RAW[:-1] #droping the last row due to noise


loads = DATA_RAW.iloc[:,3:]                                     #taking the data only
loads = loads[(np.abs(stats.zscore(loads)) < 2).all(axis=1)]    #removing outliers
loads.dropna(inplace = True)                                    #removing nan values
loads.reset_index(inplace = True)                               #reset the index
loads = loads.diff()[1:]                                        #applying first difference for removing time trend

MAX_R2 =  0
best_col =-1
best_cut = 0
col_aux = -1
loads.drop(['LMP-WEST-PJM','LMP-WESTERN-PJM','LMP-MINNESOTA-MISO','LMP-ILLINOIS-PJM','LMP-MICHIGAN-PJM','LMP-COMED-PJM','LMP-AEP-PJM','DEMAND-PJM'], axis=1, inplace=True)
aux_loads = loads.copy()
loads_orig = loads.copy()
    
for col in range(1,len(loads.columns)-1):
    print('running col:', col)
    cut = 0
    while (cut <1000):
        loads = aux_loads[cut:len(aux_loads)].copy()
        train_len = int(len(loads))
        #print('train_len : ', train_len)
        imp = loads[['IMPORT-PJM']]
        loads.drop(['IMPORT-PJM'], axis=1, inplace=True)
        regr = linear_model.LinearRegression()
        loads_train = loads[:train_len-168]
        loads_test = loads[train_len-168:]
        imp_train = imp[:train_len-168]
        imp_test = imp[train_len-168:]
        regr.fit(loads_train, imp_train)
        imp_pred = regr.predict(loads_test)
        R2 = r2_score(imp_test,imp_pred)
        if R2 > MAX_R2:
            print('col_aux:',col_aux)
            MAX_R2 = R2
            best_cut = cut
            best_col = col_aux
            print('Max R2:',MAX_R2)

        cut += 168
    col_aux = col
    aux_loads = loads_orig.drop(loads_orig.columns[col_aux],axis = 1).copy()

loads = aux_loads.drop(aux_loads.columns[best_col],axis = 1)
#print('loads: ',loads)

print('best col N: ', best_col)
print('best cut:',best_cut)
print('Max R2:',MAX_R2)

print('best col to drop: ', loads.columns[best_col])

pred_train= regr.predict(loads_train)   # predicted values for training data
residual = imp_train - pred_train       # residual in training data

#DATA_RAW = DATA_RAW.iloc[len(DATA_RAW)-1000:-1,:]
#print(DATA_RAW.index)
#A = DATA_RAW.apply(lambda x: AVGS[(AVGS.DayOfWeek==x.DayOfWeek) & (AVGS.Hour==1)]['DEMAND-MISO'],axis = 1)
#print(AVGS.loc[(AVGS.DayOfWeek=='Monday') & (AVGS.Hour==0)])
#DATA_RAW['PRUEBA'] = DATA_RAW['DayOfWeek', 'Hour'].map(AVGS.set_index(['DayOfWeek', 'Hour'])['DEMAND-MISO'])
#DATA_RAW.set_index(['DayOfWeek', 'Hour'])
#loads_train, loads_test, imp_train, imp_test = train_test_split(loads, imp, test_size=0.2, random_state=0)
#print('load_train: ',loads_test)
#print('imp_train:',imp_test)
#print(DATA_RAW.columns[1])

#plt.plot(imp)
#plt.legend()

#acf_vals = acf(residual)
#num_lags = 20
#plt.bar(range(num_lags), acf_vals[:num_lags])
#plt.show()



#elim = [4]
#loads.drop(loads.columns[elim],axis = 1, inplace = True)
