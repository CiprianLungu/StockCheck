import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import catboost as cb
import time
import pickle 
import warnings
import gc


from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, neighbors

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

from statistics import variance
from mlxtend.evaluate import bias_variance_decomp


from pathlib import Path
from os import path


warnings.filterwarnings('ignore')

#Global settings
format=10000
pretunnelLenght=10
tunnelLenght=10

maxPNL=0
maxModelPNL=0
minPNL =0
totalPNL=0
modelPNL=0

minStockToRemove = 0
stockToRemove = ""


pathGlobal = r'E:\common\stock\data_all_5min'  
file1 = open("output/removeStockLBG.txt", "a")  # append mode


# "MARA", "RIOT", "SOXL",

stockList = [ "PTON", "W", "RIG", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH","LYFT", "PENN", "SNOW", "SE", "CLF",
            "XLE", "INTC", "KVUE", "BAC",
            "C", "WBD", "PYPL", "F", "XOM", "EWZ", "GM", "AAL", "PARA", "WBA", "CCL", "PBR",
            "WFC", "VALE", "DG", "NEE", "FCX", "PINS", "ET", "DVN", "UAL",
            "LLY", "LUV", "DAL", "MRVL", "AEO" ]

#     "EEM", "EFA", "TLT", "XLP", "XLV" "SPY", "XLI", "WMT", "XLF", "DIA", "XLU", "XLB", "T", "QQQ", "TMUS", "IWM", "XLK", "PFE", "UNH", "SLV", "CMCSA" , "KR", "NKE", 

#    "SOXL", "RIOT", "MARA", "RIG", "W", "PTON", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH", "CLF", "SE", "SNOW", "PENN","LYFT", 
# 
# 
stockList = ['PTON', 'W', 'RIG', 'U', 'DKNG', 'CHWY', 'PATH', 'PLTR', 'SNAP', 'RH', 'LYFT', 'PENN', 'SNOW', 'SE', 'CLF', 'WBD', 'CCL', 'PINS', 'PARA', 'AEO', 'MRVL', 
             'DVN', 'AAL', 'UAL', 'PBR', 'FCX', 'PYPL', 'DAL', 'F', 'LUV', 'GM', 'ET', 'INTC', 'VALE', 'WBA', 'EWZ', 'WFC', 'XLE', 'C', 'KVUE', 'XOM', 'DG', 'NEE', 
             'BAC', 'LLY', 'NKE', 'KR', 'CMCSA', 'SLV', 'UNH', 'PFE', 'XLK', 'TMUS', 'T', 'XLB', 'DIA', 'WMT', 'XLI', 'XLV', 'XLP', 'TLT', 'EFA', 'EEM']

stockList = ['PTON', 'W', 'RIG', 'U', 'DKNG', 'CHWY', 'PATH', 'PLTR', 'SNAP', 'RH', 'LYFT', 'PENN', 'SNOW', 'SE', 'CLF', 'WBD', 'CCL', 'PINS', 'PARA', 'AEO', 'MRVL', 
             'DVN', 'AAL', 'UAL', 'PBR', 'FCX', 'PYPL', 'DAL', 'F', 'LUV', 'GM', 'ET', 'INTC', 'VALE', 'WBA', 'EWZ', 'WFC', 'XLE', 'C', 'KVUE', 'XOM', 'DG', 'NEE', 
             'BAC', 'LLY', 'NKE', 'KR', 'CMCSA', 'SLV', 'UNH', 'PFE', 'XLK', 'TMUS', 'T', 'XLB', 'DIA', 'WMT', 'XLI', 'XLV', 'XLP', 'TLT', 'EFA', 'EEM']

indexList=['SPY', 'IWM', 'QQQ']


#indexList=['SPY', 'IWM', 'QQQ', 'XLE', 'XLK']
#indexList=['SPY', 'QQQ', 'XLE', 'XLK']
#indexList=['SPY', 'IWM', 'QQQ', 'XLF', 'XLK', 'XLP'] XLB XLE

#indexList=['SPY', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


#Read data from directory into a dataframe and transform
def read_df(pathD, indexListCheck):


    #read calendar important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\trandingCalendar.txt'  

    cols = ['DateInput']
    cal_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    cal_df['DateInput'] = cal_df['DateInput'].str.strip()
    print(cal_df.head(10))

    cal_df['DateFormat'] = pd.to_datetime(cal_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    cal_df = cal_df.dropna().reset_index(drop=True)
    cal_df['DateString']= cal_df['DateFormat'].dt.strftime('%Y-%m-%d')
    print(cal_df.head(10))
    print(cal_df.shape)


    #read calendar central bank important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\centralBank_calendar.txt'  

    cols = ['DateInput']
    calCB_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calCB_df['DateInput'] = calCB_df['DateInput'].str.strip()
    print(calCB_df.head(10))

    calCB_df['DateFormat'] = pd.to_datetime(calCB_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calCB_df = calCB_df.dropna().reset_index(drop=True)
    calCB_df['DateString']= calCB_df['DateFormat'].dt.strftime('%Y-%m-%d')
    print(calCB_df.head(10))
    print(calCB_df.shape)


    #read calendar cconfidenx index important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\confidenceIndex_calendar.txt'  

    cols = ['DateInput']
    calCI_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calCI_df['DateInput'] = calCI_df['DateInput'].str.strip()
    print(calCI_df.head(10))

    calCI_df['DateFormat'] = pd.to_datetime(calCI_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calCI_df = calCI_df.dropna().reset_index(drop=True)
    calCI_df['DateString']= calCI_df['DateFormat'].dt.strftime('%Y-%m-%d')
    print(calCI_df.head(10))
    print(calCI_df.shape)



    #read calendar economicActibity important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\economicActivity_calendar.txt'  

    cols = ['DateInput']
    calEC_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calEC_df['DateInput'] = calEC_df['DateInput'].str.strip()
    print(calEC_df.head(10))

    calEC_df['DateFormat'] = pd.to_datetime(calEC_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calEC_df = calEC_df.dropna().reset_index(drop=True)
    calEC_df['DateString']= calEC_df['DateFormat'].dt.strftime('%Y-%m-%d')
    print(calEC_df.head(10))
    print(calEC_df.shape)


    #read calendar employment_calendar important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\employment_calendar.txt'  

    cols = ['DateInput']
    calEM_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calEM_df['DateInput'] = calEM_df['DateInput'].str.strip()
    print(calEM_df.head(10))

    calEM_df['DateFormat'] = pd.to_datetime(calEM_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calEM_df = calEM_df.dropna().reset_index(drop=True)
    calEM_df['DateString']= calEM_df['DateFormat'].dt.strftime('%Y-%m-%d')
    print(calEM_df.head(10))
    print(calEM_df.shape)





    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    #-------------------------------------------- read data
    print("Start reading files to dataframe from ", path)

    dfs = list()
    for x in range(len(stockList)):
        filePath = pathD+"\\"+stockList[x]+".txt"
        if path.exists(filePath):
            data = pd.read_csv(filePath, header=None, names=cols)
            data['File'] = stockList[x]
            dfs.append(data)
    my_df = pd.concat(dfs, ignore_index=True)
    
    
    

    print("after raw read my_df.shape=", my_df.shape)
    #print(my_df.head(10))

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)+'_orig']=my_df['Open'].shift(-i)
        my_df['High_'+str(i)+'_orig']=my_df['High'].shift(-i)
        my_df['Low_'+str(i)+'_orig']=my_df['Low'].shift(-i)
        my_df['Close_'+str(i)+'_orig']=my_df['Close'].shift(-i)
        my_df['Volume_'+str(i)+'_orig']=my_df['Volume'].shift(-i)
    my_df['Close_sale_orig']=my_df['Close'].shift(tunnelLenght)
    my_df['Date_sale_orig']=my_df['Date'].shift(tunnelLenght)


    #print(my_df.head(10))
    my_df=my_df.dropna().reset_index(drop=True)

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)]=my_df['Open_'+str(i)+'_orig']*format//my_df['Open']
        my_df['High_'+str(i)]=my_df['High_'+str(i)+'_orig']*format//my_df['Open']
        my_df['Low_'+str(i)]=my_df['Low_'+str(i)+'_orig']*format//my_df['Open']
        my_df['Close_'+str(i)]=my_df['Close_'+str(i)+'_orig']*format//my_df['Open']
        #my_df['Volume_'+str(i)]=my_df['Volume_'+str(i)+'_orig']*format//my_df['Volume']

        my_df = my_df.drop(['Open_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['High_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['Low_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['Close_'+str(i)+'_orig'], axis=1)
        #my_df = my_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

    my_df['Volume_0']=10000
    for i in range(1, pretunnelLenght):
        my_df['Volume_'+str(i)] = np.where(my_df['Volume_'+str(i)+'_orig'] > my_df['Volume_'+str(i-1)+'_orig'], my_df['Volume_'+str(i-1)]+100,my_df['Volume_'+str(i-1)]-100 )

    for i in range(0, pretunnelLenght):
        my_df = my_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

    print("after transform my_df.shape=", my_df.shape)
    #print(my_df.head(10))

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)]=my_df['Open_'+str(i)] - format
        my_df['High_'+str(i)]=my_df['High_'+str(i)] - format
        my_df['Low_'+str(i)]=my_df['Low_'+str(i)] - format
        my_df['Close_'+str(i)]=my_df['Close_'+str(i)] - format
        my_df['Volume_'+str(i)]=my_df['Volume_'+str(i)] - format


    for x in range(len(indexListCheck)):
        filePath = pathD+"\\"+indexListCheck[x]+".txt"

        dia_df = pd.read_csv(filePath, header=None, names=cols)

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+str(i)+'_orig']=dia_df['Open'].shift(-i)
            dia_df['High_'+str(i)+'_orig']=dia_df['High'].shift(-i)
            dia_df['Low_'+str(i)+'_orig']=dia_df['Low'].shift(-i)
            dia_df['Close_'+str(i)+'_orig']=dia_df['Close'].shift(-i)
            dia_df['Volume_'+str(i)+'_orig']=dia_df['Volume'].shift(-i)

        print("after raw read for stock %s has index_df.shape=",indexListCheck[x], dia_df.shape)
        #print(dia_df.head(10))
        dia_df=dia_df.dropna().reset_index(drop=True)

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexListCheck[x]+'_'+str(i)]=dia_df['Open_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['High_'+indexListCheck[x]+'_'+str(i)]=dia_df['High_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Low_'+indexListCheck[x]+'_'+str(i)]=dia_df['Low_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Close_'+indexListCheck[x]+'_'+str(i)]=dia_df['Close_'+str(i)+'_orig']*format//dia_df['Open']
            #dia_df['Volume_'+indexListCheck[x]+'_'+str(i)]=dia_df['Volume_'+str(i)+'_orig']*format//dia_df['Volume']

            dia_df = dia_df.drop(['Open_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['High_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Low_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Close_'+str(i)+'_orig'], axis=1)
            #dia_df = dia_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

        dia_df['Volume_'+indexListCheck[x]+'_0']=10000
        for i in range(1, pretunnelLenght):
            dia_df['Volume_'+indexListCheck[x]+'_'+str(i)] = np.where(dia_df['Volume_'+str(i)+'_orig'] > dia_df['Volume_'+str(i-1)+'_orig'], dia_df['Volume_'+indexListCheck[x]+'_'+str(i-1)]+100,dia_df['Volume_'+indexListCheck[x]+'_'+str(i-1)]-100 )

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexListCheck[x]+'_'+str(i)]=dia_df['Open_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['High_'+indexListCheck[x]+'_'+str(i)]=dia_df['High_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Low_'+indexListCheck[x]+'_'+str(i)]=dia_df['Low_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Close_'+indexListCheck[x]+'_'+str(i)]=dia_df['Close_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Volume_'+indexListCheck[x]+'_'+str(i)]=dia_df['Volume_'+indexListCheck[x]+'_'+str(i)] - format


        for i in range(0, pretunnelLenght):
            dia_df = dia_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

        dia_df = dia_df.drop(['Open'], axis=1)
        dia_df = dia_df.drop(['High'], axis=1)
        dia_df = dia_df.drop(['Low'], axis=1)
        dia_df = dia_df.drop(['Close'], axis=1)
        dia_df = dia_df.drop(['Volume'], axis=1)

        print("after transform dia_df.shape=", dia_df.shape)
        #print(dia_df.head(10))


        my_df = pd.merge(my_df, dia_df, on=['Date'], how='inner' )
        print("after merge dia my_df.shape=", my_df.shape)
        #print(my_df.head(10))

    #print(my_df.columns)
    #print(my_df.head(10))





    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    #my_df['target'] = np.where(my_df['PNL'] > 5, 2, np.where(my_df['PNL'] < -5, 0, 1))
    my_df['target'] = np.where(my_df['PNL'] > 0, 1, 0)





    my_df[['dateOnly', 'time']] = my_df['Date'].str.split('T', n=1, expand=True)
    #print(my_df.head(10))
    my_df[['Hour', 'Minute', 'seconds']] = my_df['time'].str.split(':', n=2, expand=True)
    my_df[['year', 'Month', 'day']] = my_df['dateOnly'].str.split('-', n=2, expand=True)
    #print(my_df.head(10))

    
    my_df['Hour'] = my_df['Hour'].astype('int')
    my_df['Minute'] = my_df['Minute'].astype('int')
    my_df['Month'] = my_df['Month'].astype('int')

    my_df['MonthOfTrimestre'] = my_df['Month'] % 3
    my_df['Trimestre'] = my_df['Month'] // 3

    my_df['DateFormat'] = pd.to_datetime(my_df['Date'], format='ISO8601')
    my_df['DayOfMonth']= my_df['DateFormat'].apply(lambda x: x.day)
    my_df['DayOfWeek']= my_df['DateFormat'].apply(lambda x: x.weekday())

    my_df['EconomicDay'] = np.where( np.isin(my_df['dateOnly'], cal_df['DateString'].values) , 1, 0)
    my_df['Centralbank'] = np.where( np.isin(my_df['dateOnly'], calCB_df['DateString'].values) , 1, 0)
    my_df['ConfidenceIndex'] = np.where( np.isin(my_df['dateOnly'], calCI_df['DateString'].values) , 1, 0)
    my_df['EconomicActivity'] = np.where( np.isin(my_df['dateOnly'], calEC_df['DateString'].values) , 1, 0)
    my_df['Employment'] = np.where( np.isin(my_df['dateOnly'], calEM_df['DateString'].values) , 1, 0)
    print(my_df)

    
    my_df.loc[( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") ),   "Hour" ] = my_df['Hour'] -1


    aggHEconomicDay = my_df.groupby(['EconomicDay']).agg(CountHours=('Date', 'count') ).reset_index()
    print(aggHEconomicDay.head(20))

    '''

    
    aggHour = my_df.groupby(['Hour']).agg(CountHours=('Date', 'count') ).reset_index()
    print(aggHour.head(20))
    aggMonth = my_df.groupby(['Month']).agg(CountMounth=('Date', 'count') ).reset_index()
    print(aggMonth.head(20))
    '''

    my_df[['dateOnly_sale_orig', 'time_sale_orig']] = my_df['Date_sale_orig'].str.split('T', n=1, expand=True)
    #print(my_df.head(10))
    print("before drop process my_df.shape=", my_df.shape)

    


    #NY Time change: 2020-03-7 2020-11-01 2021-03-13 2021-11-06 2022-03-12 2022-11-05 2023-03-10 2023-11-04

    my_df.drop(my_df[ (my_df["time"] < "16:00:00Z") & 
                    ( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") )].index , axis=0, inplace=True)
    print("drop small time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] < 15:00:00Z", my_df.shape)

    my_df.drop(my_df[ (my_df["time"] > "20:00:00Z") & 
                    ( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") )].index , axis=0, inplace=True)
    print("drop big time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] > 20:00:00Z", my_df.shape)

    my_df.drop(my_df[ (my_df["time"] < "15:00:00Z") & 
                (   ( (my_df["dateOnly"] > "2020-03-07") & (my_df["dateOnly"] < "2020-11-01") ) |
                    ( (my_df["dateOnly"] > "2021-03-13") & (my_df["dateOnly"] < "2021-11-06") ) |
                    ( (my_df["dateOnly"] > "2022-03-12") & (my_df["dateOnly"] < "2022-11-05") ) |
                    ( (my_df["dateOnly"] > "2023-03-10") & (my_df["dateOnly"] < "2023-11-04") ) )].index , axis=0, inplace=True)
    print("drop small time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] < 14:00:00Z", my_df.shape)

    my_df.drop(my_df[(my_df["time"] > "19:00:00Z") & 
                (   ( (my_df["dateOnly"] > "2020-03-07") & (my_df["dateOnly"] < "2020-11-01") ) |
                    ( (my_df["dateOnly"] > "2021-03-13") & (my_df["dateOnly"] < "2021-11-06") ) |
                    ( (my_df["dateOnly"] > "2022-03-12") & (my_df["dateOnly"] < "2022-11-05") ) |
                    ( (my_df["dateOnly"] > "2023-03-10") & (my_df["dateOnly"] < "2023-11-04") ) )].index , axis=0, inplace=True)
    print("drop big time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] > 19:00:00Z", my_df.shape)

    my_df.drop(my_df[my_df["dateOnly"] != my_df["dateOnly_sale_orig"]].index , axis=0, inplace=True)
    print("dfTest.shape after drop dateOnly!=dateOnly_sale_orig", my_df.shape)


    my_df = my_df.drop(["dateOnly_sale_orig"], axis=1)
    my_df = my_df.drop(["Date_sale_orig"], axis=1)
    #my_df = my_df.drop(["Date"], axis=1)
    my_df = my_df.drop(["Open"], axis=1)
    my_df = my_df.drop(["High"], axis=1)
    my_df = my_df.drop(["Low"], axis=1)
    my_df = my_df.drop(["Close"], axis=1)
    my_df = my_df.drop(["Close_sale_orig"], axis=1)
    my_df = my_df.drop(["year"], axis=1)
    my_df = my_df.drop(["day"], axis=1)

    ''' 
    aggHour2 = my_df.groupby(['Hour']).agg(CountHours=('Month', 'count') ).reset_index()
    print(aggHour2.head(20))
    aggMonth2 = my_df.groupby(['Month']).agg(CountMounth=('Hour', 'count') ).reset_index()
    print(aggMonth2.head(20))
    '''


    my_df = my_df.reset_index() 
    my_df = my_df.dropna().reset_index(drop=True)

    #print(my_df)

    print("Finish reading files to dataframe from ", path)
    return my_df




def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    global stockList
    global my_dfGAll
    


    dfTestFeat = dfTest[colsFeat]
    forecasts = model.predict_proba(dfTestFeat)


    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecastSell", "forecastBuy"]) ], axis=1)

    result['forecast'] = np.where(result['forecastSell'] > 0.5, result['forecastSell'], result['forecastBuy'])
    result['action'] = np.where(result['forecastSell'] > 0.5, 'sell', 'buy')
    result['pnlAbs'] = abs(result['PNL'])


    
    for k in range(len(result)):
        if(result["forecastSell"].iloc[k]> 0.5):
            result["PNL"].iloc[k] = - result["PNL"].iloc[k]


    time_df = result.groupby(["Date"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
    time_df[ "PNLPerTrade"] =  time_df["PNL"] / time_df["Trades"]
    #time_df.to_csv('E:\Work\GitHub\Energy\output/my_dfdatetime.txt', sep=';', index=True, mode='a' )

    totalBuy = 0
    meanBuy = 0
    noGain = 0
    noLost = 0
    noBuy = 0
    noLostGlobal = 0
    noGainGlobal = 0
    for k in range(len(time_df)):
        #print(time_df["Date"].iloc[k])
        timeResult = result[(result["Date"] == time_df["Date"].iloc[k])]
        timeResult = timeResult.sort_values(by=['forecast'], ascending=False)

        for p in range(len(timeResult)):
            if(timeResult["PNL"].iloc[p] > 0):
                noGainGlobal = noGainGlobal +1
            else:
                noLostGlobal = noLostGlobal +1

        if(len(timeResult) < 5):
            continue

        for p in range(2):
            totalBuy = totalBuy + timeResult["PNL"].iloc[p]
            if(timeResult["PNL"].iloc[p] > 0):
                noGain = noGain +1
            else:
                noLost = noLost +1
            noBuy = noBuy +1

    meanBuy = totalBuy /noBuy
    print(str(no)+". TotalBuy="+str( round(totalBuy,2) )+" meanBuy="+str(meanBuy)+" acuracy="+str(noGain/noBuy)+" acuracyGlobal="+str(noGainGlobal/(noGainGlobal+noLostGlobal)))
    totalPNL = totalPNL +totalBuy


    my_dfG = result.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
    my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
    my_dfG = my_dfG.sort_values(by=['PNLPerTrade'], ascending=True)


    my_dfGAll = pd.concat([my_dfGAll, my_dfG], ignore_index=True)





# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================







dateList = [ "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", "2021-12-01", 
            "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01", 
            "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"]



indexListCheck = indexList

colsFeat = []

for i in range(0, pretunnelLenght):
    colsFeat.append('Open_'+str(i))
    colsFeat.append('High_'+str(i))
    colsFeat.append('Low_'+str(i))
    colsFeat.append('Close_'+str(i))
    colsFeat.append('Volume_'+str(i))

for x in indexListCheck:
    for i in range(0, pretunnelLenght):
        colsFeat.append('Open_'+x+'_'+str(i))
        colsFeat.append('High_'+x+'_'+str(i))
        colsFeat.append('Low_'+x+'_'+str(i))
        colsFeat.append('Close_'+x+'_'+str(i))
        colsFeat.append('Volume_'+x+'_'+str(i))
colsFeat.append('Month')
colsFeat.append('DayOfMonth')
colsFeat.append('DayOfWeek')
colsFeat.append('Hour')
colsFeat.append('Minute')
#colsFeat.append('EconomicDay')
#colsFeat.append('Month')
#colsFeat.append('MonthOfTrimestre')
#colsFeat.append('Trimestre')
 
#colsFeat.append('Centralbank')
#colsFeat.append('ConfidenceIndex')
#colsFeat.append('EconomicActivity')
#colsFeat.append('Employment')



#Read data
dfData=read_df(pathGlobal, indexListCheck)



for k in range(50):
    print("------------------ START --------------------------------")
    print("current len(stockList="+str(len(stockList)))

    minStockToRemove = 0
    stockToRemove = ""
    modelPNL = 0
    maxPNL =0
    minPNL = 10000

    modelPNLBase = 0
    maxPNLBase =0
    minPNLBase = 10000

    my_dfGAll  = pd.DataFrame()

    for k in range(0, 19):




        dateStartTraning = dateList[k]
        dateEndTraning = dateList[k+12*2]

        dateStartTest = dateList[k+12*2]
        dateEndTest = dateList[k+12*2+1]

        print("===>> Check mounth : Traning : "+str(dateStartTraning)+" --> "+str(dateEndTraning)+"  Testing : "+str(dateStartTest)+" --> "+str(dateEndTest))
        file1.write("===>> Check mounth : Traning : "+str(dateStartTraning)+" --> "+str(dateEndTraning)+"  Testing : "+str(dateStartTest)+" --> "+str(dateEndTest)+ " \n")

        #dfFit1 = dfData[(dfData['Date'] >= dateStartTraning) & (dfData['Date'] < dateEndTraning) ]
        dfFit1 = dfData[ (dfData['Date'] < dateEndTraning) ]
        dfTest1 = dfData[(dfData['Date'] >= dateStartTest) & (dfData['Date'] < dateEndTest) ]



        dfFit1 = dfFit1.sort_values(by=['Date'], ascending=True)
        dfTest1 = dfTest1.sort_values(by=['Date'], ascending=True)

        dfFit1 = dfFit1.reset_index() 
        dfTest1 = dfTest1.reset_index() 

        ''' 
        print("Variance of dfFit1 is % s " %(variance(dfFit1['PNL'])))
        print("Variance of dfTest1 is % s " %(variance(dfTest1['PNL'])))

        print("Mean np of dfFit1: ", np.mean(dfFit1['PNL'])) 
        print("Mean np of dfTest1: ", np.mean(dfTest1['PNL']))

        print("STD np of dfFit1: ", np.std(dfFit1['PNL'])) 
        print("STD np of dfTest1: ", np.std(dfTest1['PNL']))

        print("variance np of dfFit1: ", np.var(dfFit1['PNL'])) 
        print("variance np of dfTest1: ", np.var(dfTest1['PNL']))
        '''



        features = dfFit1[colsFeat]
        target = dfFit1['target']

        scaler = StandardScaler()
        #features = scaler.fit_transform(features)
        #dfTest1= scaler.transform(dfTest1)
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            features, target, test_size=0.1, random_state=1, shuffle=False )
        #test_size=0.1, random_state=104, shuffle=True
        #print("X= ", X_train.shape, X_valid.shape)
        #print("Y= ", Y_train.shape, Y_valid.shape)

        #model =  LGBMClassifier(verbose=0, feature_fraction= 0.7)
        #model = CatBoostClassifier(verbose=0)
        
        #
        #

        model = LGBMClassifier(verbose=0)
        model2 = LGBMClassifier(verbose=0, learning_rate=0.005, n_estimators=1000)

        


        #model2 = CatBoostClassifier(verbose=0, learning_rate=0.08, iterations=2000, depth=8)
        #model2 = LGBMClassifier(verbose=0, learning_rate=0.07, n_estimators=10000, min_child_samples=5, min_child_weight=0.001)
        #model2 = LGBMClassifier(verbose=0, learning_rate=0.001, n_estimators=10000 )

        #model2 = CatBoostClassifier(verbose=0)

        
        #model2 =  LGBMClassifier(verbose=0, learning_rate=0.001, n_estimators=50000)
        #model = MLPClassifier(activation='identity', max_iter=5000, hidden_layer_sizes=(200,130,130,200), learning_rate_init=0.0001, random_state=0)

        #model = MLPClassifier(activation='identity', max_iter=1000, hidden_layer_sizes=(150,100,150), learning_rate_init=0.001, random_state=0)
        #model2 = MLPClassifier(activation='identity', max_iter=35000, hidden_layer_sizes=(200,150,100,150), learning_rate_init=0.0001, random_state=0)

        print('clungu start training no 1 at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
        model.fit(X_train, Y_train)

        print('Train Acc: : '+str(metrics.roc_auc_score(Y_train, model.predict(X_train)))+'  Valid Acc : '+str(metrics.roc_auc_score(Y_valid, model.predict(X_valid))))
        print('Log Lost : ', metrics.log_loss(Y_valid, model.predict(X_valid)))    
        #print('Validation Score : ', model.score(X_valid, Y_valid) )

        file1.write('Train Acc: : '+str(metrics.roc_auc_score(Y_train, model.predict(X_train)))+'  Valid Acc : '+str(metrics.roc_auc_score(Y_valid, model.predict(X_valid)))+ " \n")


        #y_error, avg_bias, avg_var = bias_variance_decomp(model, 
        #                                          X_train.values, Y_train.values,
        #                                          X_valid.values, Y_valid.values, 
        #                                          loss='0-1_loss', 
        #                                          random_seed=23)

        # Display feature importance with feature names
        #gain_importance = model.feature_importances_
        #feature_names = colsFeat
        #gain_importance_df = pd.DataFrame({'Feature': feature_names, 'Gain': gain_importance})
        #print(gain_importance_df.sort_values(by='Gain', ascending=False).to_string())

        totalPNL=0
        checkForecast(dfTest1, model, 0)

        if maxPNLBase < totalPNL:
            maxPNLBase = totalPNL
        if minPNLBase > totalPNL:
            minPNLBase = totalPNL    
        modelPNLBase = modelPNLBase + totalPNL


        print('clungu start training no 2 at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
        model2.fit(X_train, Y_train)
        
        print('Train Acc: : '+str(metrics.roc_auc_score(Y_train, model2.predict(X_train)))+'  Valid Acc : '+str(metrics.roc_auc_score(Y_valid, model2.predict(X_valid))))
        print('Log Lost : ', metrics.log_loss(Y_valid, model2.predict(X_valid)))    
        #print('Validation Score : ', model2.score(X_valid, Y_valid) )

        file1.write('Train Acc: : '+str(metrics.roc_auc_score(Y_train, model2.predict(X_train)))+'  Valid Acc : '+str(metrics.roc_auc_score(Y_valid, model2.predict(X_valid)))+ " \n")

        # Display feature importance with feature names
        #gain_importance = model2.feature_importances_
        #feature_names = colsFeat
        #gain_importance_df = pd.DataFrame({'Feature': feature_names, 'Gain': gain_importance})
        #print(gain_importance_df.sort_values(by='Gain', ascending=False).to_string())

        totalPNL=0
        checkForecast(dfTest1, model2, 0)


        if maxPNL < totalPNL:
            maxPNL = totalPNL
        if minPNL > totalPNL:
            minPNL = totalPNL    
        modelPNL = modelPNL + totalPNL

        

        print("==> ModelPNLBase="+str(modelPNLBase)+" while current maxPNLBase="+str(maxPNLBase)+" and current minPNLBase="+str(minPNLBase))
        print("==> ModelPNLParameter="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL))
        #print("----------------------------------------------------------------------")

        file1.write("==> ModelPNLBase="+str(modelPNLBase)+" while current maxPNLBase="+str(maxPNLBase)+" and current minPNLBase="+str(minPNLBase)+ " \n")
        file1.write("==> ModelPNLParameter="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL)+ " \n")






        del dfFit1
        del dfTest1


        gc.collect()

    if(modelPNLBase > maxModelPNL):
        maxModelPNL = modelPNLBase
    print("-------------------MODEL DONE  ----------------------------------") 
    print("----------------------------------------------------------------------") 
    print("==> Total MODEL PNL="+str(modelPNLBase)+" while current maxPNL="+str(maxPNLBase)+" and current minPNL="+str(minPNLBase)+" and current maxModelPNL="+str(maxModelPNL))
    print("----------------------------------------------------------------------") 

    file1.write("-------------------MODEL DONE  ----------------------------------"+ " \n")
    file1.write("==> Total MODEL PNL="+str(modelPNLBase)+" while current maxPNL="+str(maxPNLBase)+" and current minPNL="+str(minPNLBase)+" and current maxModelPNL="+str(maxModelPNL)+ " \n")

    my_dfGAll = my_dfGAll.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('Trades', 'sum') , pnlAbsM=('pnlAbsM', 'mean')  ).reset_index()
    my_dfGAll[ "PNLPerTrade"] =  my_dfGAll["PNL"] / my_dfGAll["Trades"]
    my_dfGAll = my_dfGAll.sort_values(by=['PNLPerTrade'], ascending=True)

    print( my_dfGAll)
    stockToRemove = my_dfGAll["File"].iloc[0]
    print("use stockToRemove=", stockToRemove)
    
    print("before remove stock dfData.shape=", dfData.shape)
    dfData.drop(dfData[ (dfData["File"] == stockToRemove)].index , axis=0, inplace=True)
    stockList.remove(str(stockToRemove))
    print("after remove stock dfData.shape=", dfData.shape)
    print( stockList)
    file1.write(str(stockList)+ " \n")
