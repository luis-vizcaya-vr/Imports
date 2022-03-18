# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:18:58 2022

@author: I34896
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:/Users/I34896/Desktop/TEMPERATURE_FORECASTS.csv')

#dataframe with all temperature forecasts and fuel prices
dataFrame = pd.DataFrame(data, columns=['WSI: Forecast Temp JONESBORO AR (JBR)(64720)', 'WSI: Forecast Temp LITTLE ROCK AR (LIT)(64721)', 'WSI: Forecast Temp NEW ORLEANS LA (MSY)(64723)', 
                                        'WSI: Forecast Temp NEW ORLEANS LA (MSY)(64723)', 'WSI: Forecast Temp MONROE LA (MLU)(64724)', 'WSI: Forecast Temp JACKSON MS (JAN)(64725)'
                                        , 'WSI: Forecast Temp PORT ARTHUR TX (BPT)(64727)', 'WSI: Forecast Temp HUNTSVILLE TX (UTS)(64728)', 'WSI: Forecast Temp BUFFALO NY (KBUF)(64988)', 
                                        'WSI: Forecast Temp JFK NY (KJFK)(64990)', 'WSI: Forecast Temp Long Island NY (KISP)(64996)', 'WSI: Forecast Temp DC (KDCA)(65008)', 'WSI: Forecast Temp NORWOOD MA (OWD)(65064)'
                                        , 'WSI: Forecast Temp MANASSAS VA (HEF)(65067)', 'WSI: Forecast Temp NEWBURG NY (SWF)(65069)', 'WSI: Forecast Temp Houston (G.B. Int’l) TX (IAH)(65122)', 
                                        'WSI: Forecast Temp Dallas – Fort Worth TX  (DFW)(65124)', 'WSI: Forecast Temp San Antonio TX  (SAT)(65126)', 'WSI: Forecast Temp Austin TX  (AUS)(65128)', 'WSI: Forecast Temp Corpus Christi TX  (CRP)(65130)',
                                        'WSI: Forecast Temp McAllen TX  (MFE)(65132)', 'WSI: Forecast Temp Midland TX  (MAF)(65134)', 'WSI: Forecast Temp Abilene TX  (ABI)(65136)', 'WSI: Forecast Temp Fullerton/Anaheim CA  (FUL)(65138)', 
                                        'WSI: Forecast Temp SAN JOSE CA  (SJC)(65141)', 'WSI: Forecast Temp BOISE ID (BOI)(65820)', 'WSI: Forecast Temp MEDFORD OR (MFR)(65822)', 'WSI: Forecast Temp PORTLAND OR (PDX)(65824)', 'WSI: Forecast Temp SEATTLE WA (SEA)(65826)',
                                        'WSI: Forecast Temp SPOKANE/METRO WA (GEG)(65928)', 'WSI: Forecast Temp FLAGSTAFF AZ (FLG)(65996)', 'WSI: Forecast Temp PHOENIX AZ (PHX)(65998)', 'WSI: Forecast Temp RIVERSIDE CA (RIV)(66000)', 'WSI: Forecast Temp CONCORD NH (CON)(66002)',
                                        'WSI: Forecast Temp Manchester NH(MHT)(66132)', 'WSI: Forecast Temp Bridgeport CT(BDR)(66134)', 'WSI: Forecast Temp Taunton MA(TAN)(66136)', 'WSI: Forecast Temp Worchester MA(ORH)(66138)', 'WSI: Forecast Temp Burlington VT(BTV)(66140)', 
                                        'WSI: Forecast Temp Portland ME(PWM)(66142)', 'WSI: Forecast Temp Westfield MA(BAF)(66144)', 'WSI: Forecast Temp White Plains NY(HPN)(66146)', 'WSI: Forecast Temp Plattsburgh NY (PBG)(66148)', 'WSI: Forecast Temp Rome NY(RME)(66150)', 
                                        'WSI: Forecast Temp Kansas City MO (LXT)(71275)', 'WSI: Forecasted Temp Salinas CA (SNS )(78059)', 'WSI: Forecasted Temp Redding CA (RDD )(78061)', 'WSI: Forecasted Temp San Luis Obispo CA (SBP )(78063)', 'WSI: Forecasted Temp Santa Barbara CA (SBA )(78065)',
                                        'WSI: Forecasted Temp Dagget CA (DAG )(78067)', 'WSI: Forecasted Temp Edwards Air Force Base CA (EDW )(78069)', 'NORM_1', 'NORM_2', 'NORM_3', 'NORM_4', 'NORM_5', 'NORM_6', 'NORM_7', 'NORM_8', 'NORM_9'])

# form correlation matrix
matrix = dataFrame.corr()

#dictionary that connects fuel prices with the most correlated temperature sensor
Fuel_Temper_Dict = {
    "Dominion South": 66000,
    "Transco-Z6 Non-NY": 78065,
    "Tetco M-3": 78063,
    "Northern Appalachia": 78063,
    "Central Appalachia": 78059,
    "Chicago": 66000,
    "Transco-Z5 (non-WGL)": 78063,
    "Henry": 66000,
    "Transco Z6 (NY)": 78063
}


