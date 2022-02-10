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
print(DATA_RAW)
DATA_RAW.dropna(inplace=True)
DATA = DATA_RAW.groupby(['DayOfWeek'],as_index = False).mean() #calculating average for each hour 

AVG2 = pd.DataFrame()
STD2 = pd.DataFrame()
plt.plot(DATA[['IMPORT-PJM']])
plt.show()


DATA_RAW = DATA_RAW[['IMPORT-PJM']]
AVGS = DATA_RAW.groupby(['DayOfWeek']).mean() #calculating average for each hour 
AVGS.reset_index(inplace=True)                                                      #reset the index
STDVS = DATA_RAW.groupby(['DayOfWeek']).std()                   #calculating std for each hour
STDVS.reset_index(inplace=True)   
