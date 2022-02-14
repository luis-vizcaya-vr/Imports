import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error


DATA_RAW = pd.read_csv('Import_Data_daily.csv')
DATA_RAW.dropna(inplace=True)
DATA_RAW.reset_index()

def Hyper_Param_Sarima(Data, ps,qs,Ps,Qs,Ss,starts,Forecast_Period = 7):
    MIN_MAE = 5000000
    MIN_MAX_ERROR =5000000
    results={}
    Data_len = int(len(Data))
    
    for start in starts:
        Data_train = Data[start:Data_len-Forecast_Period]
        Data_test = Data[Data_len-Forecast_Period:]

        for p in ps:
            for q in qs:
                for P in Ps:
                    for Q in Qs:
                        for S in Ss:
                            print('PARAMS:',[p,q,P,Q,S,start])
                            """ my_order = (p,1,q)
                            my_seasonal_order = (P, 0, Q, S)
                            model = SARIMAX(Data_train, order=my_order, seasonal_order=my_seasonal_order)
                            model_fit = model.fit(disp = False)
                            predictions = model_fit.forecast(Forecast_Period)
                            predictions = pd.Series(predictions)
                            predictions.reset_index(inplace = True, drop=True)
                            Data_test.reset_index(inplace = True, drop=True)
                             """
                            R = Evaluate_Sarima(Data_train, Data_test, p, 1, q, P, 0, Q, S)
                            MAE = R['MAE']
                            MAX_ERROR = R['MAX_ERROR']

                            if MAX_ERROR < MIN_MAX_ERROR:
                                MIN_MAX_ERROR = MAX_ERROR
                                MIN_MAE = MAE
                                Best_Par = [p,q,P,Q,S,start] 

    results['MIN_MAE'] = MIN_MAE
    results['MIN_MAX_MAE'] = MIN_MAX_ERROR
    results['Best_Par'] = Best_Par
    return results

def Evaluate_Sarima(Data_train, Data_test, p, d, q, P, D, Q, S, Forecast_Period = 7):
    results = {}
    my_order = (p,d,q)
    my_seasonal_order = (P, D, Q, S)
    model = SARIMAX(Data_train, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit(disp = False)
    predictions = model_fit.forecast(Forecast_Period)
    predictions = pd.Series(predictions)
    predictions.reset_index(inplace = True, drop=True)
    Data_test.reset_index(inplace = True, drop=True)
    MAE = mean_absolute_error(Data_test, predictions)
    MAX_ERROR = max_error(Data_test, predictions)
    results['MAE'] = MAE
    results['MAX_ERROR'] = MAX_ERROR
    results['Forecast'] = predictions
    return results
 
                            
