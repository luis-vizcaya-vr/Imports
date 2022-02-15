from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import max_error, mean_absolute_error


DATA_RAW = pd.read_csv('Import_Data_v6.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW['TimeLabel'] = DATA_RAW['DayOfWeek'].apply(str) + DATA_RAW['Hour'].apply(str)
DATA_RAW['TimeLabel'].to_csv('timelabel_data_raw.csv')

AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
AVGS = DATA_RAW.groupby(['DayOfWeek', 'Hour'], as_index = False).mean() #calculating average for each hour 
AVGS.reset_index()                                                      #reset the index
AVGS['TimeLabel'] = AVGS['DayOfWeek'].apply(str) + AVGS['Hour'].apply(str)
STDVS = DATA_RAW.groupby(['DayOfWeek', 'Hour']).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True)                
STDVS['TimeLabel'] = STDVS['DayOfWeek'].apply(str) + STDVS['Hour'].apply(str)

for c in DATA_RAW.columns[3:-3]:
    AVG2[c] = DATA_RAW['TimeLabel'].map(AVGS.set_index('TimeLabel')[c])
    STD2[c] = DATA_RAW['TimeLabel'].map(STDVS.set_index('TimeLabel')[c])
AVG2['TimeLabel'] = DATA_RAW['TimeLabel']
STD2['TimeLabel'] = DATA_RAW['TimeLabel']
 
DATA_RAW = DATA_RAW[:-1] #droping the last row due to noise
loads = DATA_RAW.iloc[:,3:]                                     #taking the data only
loads.drop(['TimeLabel'], axis=1, inplace=True)
print('loads:',loads)

loads = loads[(np.abs(stats.zscore(loads)) < 2).all(axis=1)]    #removing outliers
loads.dropna(inplace = True)                                    #removing nan values
loads = loads.diff()[1:]                                        #applying first difference for removing time trend
loads.reset_index(inplace = True)                               #reset the index
        
MAX_R2 =  0
MAX_MAE = 5000000
MAX_MAX_ERROR = 5000000
best_col =-1
best_cut = 0
col_aux = -1
loads.drop(['IMPORT-NY'], axis=1, inplace=True)
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
        MAE = mean_absolute_error(imp_test, imp_pred)
        MAE = max_error(imp_test, imp_pred)
        
        if MAX_MAE > MAE:
            print('col_aux:',col_aux)
            MAX_MAE = MAE
            best_cut = cut
            best_col = col_aux
            print('Max R2:',MAX_MAE)
            
        cut += 168
    col_aux = col
    aux_loads = loads_orig.drop(loads_orig.columns[col_aux],axis = 1).copy()

if best_col >-1:
    loads = loads_orig.drop(loads_orig.columns[best_col],axis = 1)
    print('best col to drop: ', loads.columns[best_col])

print('best col N: ', best_col)
print('best cut:',best_cut)
print('Max R2:',MAX_R2)
print('MAE:', MAX_MAE)


#pred_train= regr.predict(loads_train)   
#residual = imp_train - pred_train       
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
#imp_pred.reset_inde(inplace = True, Drop = True)
imp_test.reset_index(inplace = True, drop = True)

plt.plot(imp_pred)
plt.plot(imp_test)
#plt.legend()

#acf_vals = acf(residual)
#num_lags = 20
#plt.bar(range(num_lags), acf_vals[:num_lags])
plt.show()



#elim = [4]
#loads.drop(loads.columns[elim],axis = 1, inplace = True)
