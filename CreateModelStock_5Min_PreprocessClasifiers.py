import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import catboost as cb
import time
import pickle 
import warnings

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, neighbors

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


from pathlib import Path
from os import path


warnings.filterwarnings('ignore')

#Global settings
format=10000
pretunnelLenght=10
tunnelLenght=10

maxPNL=0
totalPNL=0


pathGlobal = r'E:\common\stock\data_all_5min'  


#"MARA", "RIOT", "SOXL",  "XLU" , "XLF"
stockList = [ "PTON", "W", "RIG", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH","LYFT", "PENN", "SNOW", "SE", "CLF",
            "XLE", "INTC", "KVUE", "BAC",
            "C", "WBD", "PYPL", "F", "XOM", "EWZ", "GM", "AAL", "PARA", "WBA", "CCL", "PBR",
            "WFC", "VALE", "DG", "NEE", "FCX", "PINS", "DVN", "UAL",
            "LLY", "LUV", "DAL", "MRVL", "AEO" ]

 
stockList = [ "EEM","EFA","TLT","XLP","XLV","XLI","WMT","DIA","XLB","T","TMUS","XLK","PFE","UNH","SLV","CMCSA",
            "KR","NKE","LLY","BAC","NEE","DG","XOM","KVUE","C","XLE","WFC","EWZ","WBA","VALE","INTC","ET","GM","LUV","F","DAL","PYPL","FCX","PBR",
            "UAL","AAL","DVN","MRVL","AEO","PARA","PINS","CCL","WBD","CLF","SE","SNOW","PENN","LYFT","RH","SNAP","PLTR","PATH","CHWY","DKNG","U","RIG","W","PTON"]

stockList = ['PTON', 'W', 'RIG', 'U', 'DKNG', 'CHWY', 'PATH', 'PLTR', 'SNAP', 'RH', 'LYFT', 'PENN', 'SNOW', 'SE', 'CLF', 'WBD', 'CCL', 'PINS', 'PARA', 'AEO', 'MRVL', 
             'DVN', 'AAL', 'UAL', 'PBR', 'FCX', 'PYPL', 'DAL', 'F', 'LUV', 'GM', 'ET', 'INTC', 'VALE', 'WBA', 'EWZ', 'WFC', 'XLE', 'C', 'KVUE', 'XOM', 'DG', 'NEE', 
             'BAC', 'LLY', 'NKE', 'KR', 'CMCSA', 'SLV', 'UNH', 'PFE', 'XLK', 'TMUS', 'T', 'XLB', 'DIA', 'WMT', 'XLI', 'XLV', 'XLP', 'TLT', 'EFA', 'EEM']

indexList=['SPY', 'IWM', 'QQQ']



def read_df(pathD, indexListCheck):
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
        my_df['Open_'+str(i)]=my_df['Open_'+str(i)] - format
        my_df['High_'+str(i)]=my_df['High_'+str(i)] - format
        my_df['Low_'+str(i)]=my_df['Low_'+str(i)] - format
        my_df['Close_'+str(i)]=my_df['Close_'+str(i)] - format
        my_df['Volume_'+str(i)]=my_df['Volume_'+str(i)] - format
    

    print("after transform my_df.shape=", my_df.shape)
    print(my_df.head(3))


    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    epsilon = 20
    my_df['target'] = np.where(my_df['PNL'] > epsilon, 2, np.where(my_df['PNL'] < -epsilon, 0, 1))
    #my_df['target'] = np.where(my_df['PNL'] > 0, 1, 0)


    my_df[['dateOnly', 'time']] = my_df['Date'].str.split('T', n=1, expand=True)
    my_df[['Hour', 'Minute', 'seconds']] = my_df['time'].str.split(':', n=2, expand=True)
    my_df[['year', 'Month', 'day']] = my_df['dateOnly'].str.split('-', n=2, expand=True)

    my_df['Hour'] = my_df['Hour'].astype('int')
    my_df['Minute'] = my_df['Minute'].astype('int')
    my_df.loc[( (my_df["dateOnly"] < "2020-03-07") |
                ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                ( my_df["dateOnly"] > "2023-11-04") ),   "Hour" ] = my_df['Hour'] -1
    my_df['Month'] = my_df['Month'].astype('int')

    #my_df['MonthOfTrimestre'] = my_df['Month'] % 3
    #my_df['Trimestre'] = my_df['Month'] // 3

    my_df['DateFormat'] = pd.to_datetime(my_df['Date'], format='ISO8601')
    my_df['DayOfMonth']= my_df['DateFormat'].apply(lambda x: x.day)
    my_df['DayOfWeek']= my_df['DateFormat'].apply(lambda x: x.weekday())


    
    #my_df = my_df.drop(["Date"], axis=1)
    my_df = my_df.drop(["Open"], axis=1)
    my_df = my_df.drop(["High"], axis=1)
    my_df = my_df.drop(["Low"], axis=1)
    my_df = my_df.drop(["Close"], axis=1)
    my_df = my_df.drop(["Close_sale_orig"], axis=1)
    my_df = my_df.drop(["year"], axis=1)
    my_df = my_df.drop(["day"], axis=1)

    print(my_df.head(3))




    #merrge index data
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
            dia_df = dia_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

        
        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexListCheck[x]+'_'+str(i)]=dia_df['Open_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['High_'+indexListCheck[x]+'_'+str(i)]=dia_df['High_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Low_'+indexListCheck[x]+'_'+str(i)]=dia_df['Low_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Close_'+indexListCheck[x]+'_'+str(i)]=dia_df['Close_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Volume_'+indexListCheck[x]+'_'+str(i)]=dia_df['Volume_'+indexListCheck[x]+'_'+str(i)] - format
        

        dia_df = dia_df.drop(['Open'], axis=1)
        dia_df = dia_df.drop(['High'], axis=1)
        dia_df = dia_df.drop(['Low'], axis=1)
        dia_df = dia_df.drop(['Close'], axis=1)
        dia_df = dia_df.drop(['Volume'], axis=1)

        print("after transform dia_df.shape=", dia_df.shape)
        #print(dia_df.head(3))


        my_df = pd.merge(my_df, dia_df, on=['Date'], how='inner' )
        print("after merge dia my_df.shape=", my_df.shape)
        #print(my_df.head(3))
        #print(my_df.columns)

    #print(my_df.columns)
    #print(my_df.head(10))





    



    '''

    aggHour = my_df.groupby(['Hour']).agg(CountHours=('Date', 'count') ).reset_index()
    print(aggHour.head(20))
    aggMonth = my_df.groupby(['Month']).agg(CountMounth=('Date', 'count') ).reset_index()
    print(aggMonth.head(20))
    '''

    my_df[['dateOnly_sale_orig', 'time_sale_orig']] = my_df['Date_sale_orig'].str.split('T', n=1, expand=True)
    #print(my_df.head(10))
    print("before drop process my_df.shape=", my_df.shape)

    

    nowS = datetime.datetime.now()
    nowS = nowS.replace(hour=14, minute=40)
    nowSRemove = nowS + datetime.timedelta(minutes=5*pretunnelLenght)
    print("nowSRemove="+nowSRemove.strftime('%Y-%m-%dT%H:%M:%SZ'))

    nowE = datetime.datetime.now()
    nowE = nowE.replace(hour=21, minute=00)
    nowERemove = nowE - datetime.timedelta(minutes=5*tunnelLenght)
    print("nowERemove="+nowERemove.strftime('%Y-%m-%dT%H:%M:%SZ'))


    #NY Time change: 2020-03-7 2020-11-01 2021-03-13 2021-11-06 2022-03-12 2022-11-05 2023-03-10 2023-11-04

    my_df.drop(my_df[ (my_df["time"] < nowSRemove.strftime('%H:%M:%SZ')) & 
                    ( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") )].index , axis=0, inplace=True)
    print("drop small time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] < " +nowSRemove.strftime('%H:%M:%SZ')+ " has shape=" + str(my_df.shape))

    my_df.drop(my_df[ (my_df["time"] > nowERemove.strftime('%H:%M:%SZ')) & 
                    ( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") )].index , axis=0, inplace=True)
    print("drop big time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] > " +nowERemove.strftime('%H:%M:%SZ')+ " has shape=" + str(my_df.shape))

    nowS = datetime.datetime.now()
    nowS = nowS.replace(hour=13, minute=40)
    nowSRemove = nowS + datetime.timedelta(minutes=5*pretunnelLenght)
    print("nowSRemove="+nowSRemove.strftime('%Y-%m-%dT%H:%M:%SZ'))

    nowE = datetime.datetime.now()
    nowE = nowE.replace(hour=20, minute=00)
    nowERemove = nowE - datetime.timedelta(minutes=5*tunnelLenght)
    print("nowERemove="+nowERemove.strftime('%Y-%m-%dT%H:%M:%SZ'))

    my_df.drop(my_df[ (my_df["time"] < nowSRemove.strftime('%H:%M:%SZ')) & 
                (   ( (my_df["dateOnly"] > "2020-03-07") & (my_df["dateOnly"] < "2020-11-01") ) |
                    ( (my_df["dateOnly"] > "2021-03-13") & (my_df["dateOnly"] < "2021-11-06") ) |
                    ( (my_df["dateOnly"] > "2022-03-12") & (my_df["dateOnly"] < "2022-11-05") ) |
                    ( (my_df["dateOnly"] > "2023-03-10") & (my_df["dateOnly"] < "2023-11-04") ) )].index , axis=0, inplace=True)
    print("drop small time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] < " +nowSRemove.strftime('%H:%M:%SZ')+ " has shape=" + str(my_df.shape))

    my_df.drop(my_df[(my_df["time"] > nowERemove.strftime('%H:%M:%SZ')) & 
                (   ( (my_df["dateOnly"] > "2020-03-07") & (my_df["dateOnly"] < "2020-11-01") ) |
                    ( (my_df["dateOnly"] > "2021-03-13") & (my_df["dateOnly"] < "2021-11-06") ) |
                    ( (my_df["dateOnly"] > "2022-03-12") & (my_df["dateOnly"] < "2022-11-05") ) |
                    ( (my_df["dateOnly"] > "2023-03-10") & (my_df["dateOnly"] < "2023-11-04") ) )].index , axis=0, inplace=True)
    print("drop big time my_df.shape=", my_df.shape)
    print("dfTest.shape after drop my_df[time] > " +nowERemove.strftime('%H:%M:%SZ')+ " has shape=" + str(my_df.shape))

    my_df.drop(my_df[my_df["dateOnly"] != my_df["dateOnly_sale_orig"]].index , axis=0, inplace=True)
    print("dfTest.shape after drop dateOnly!=dateOnly_sale_orig", my_df.shape)




    ''' 
    aggHour2 = my_df.groupby(['Hour']).agg(CountHours=('Month', 'count') ).reset_index()
    print(aggHour2.head(20))
    aggMonth2 = my_df.groupby(['Month']).agg(CountMounth=('Hour', 'count') ).reset_index()
    print(aggMonth2.head(20))
    '''

    my_df = my_df.drop(["dateOnly_sale_orig"], axis=1)
    my_df = my_df.drop(["Date_sale_orig"], axis=1)






    #read calendar central bank important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\centralBank_calendar.txt'  
    cols = ['DateInput']
    calCB_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calCB_df['DateInput'] = calCB_df['DateInput'].str.strip()
    #print(calCB_df.head(10))
    calCB_df['DateFormat'] = pd.to_datetime(calCB_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calCB_df = calCB_df.dropna().reset_index(drop=True)
    calCB_df['DateString']= calCB_df['DateFormat'].dt.strftime('%Y-%m-%d')
    #print(calCB_df.head(10))
    print(calCB_df.shape)


    #read calendar cconfidenx index important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\confidenceIndex_calendar.txt'  
    cols = ['DateInput']
    calCI_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calCI_df['DateInput'] = calCI_df['DateInput'].str.strip()
    #print(calCI_df.head(10))
    calCI_df['DateFormat'] = pd.to_datetime(calCI_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calCI_df = calCI_df.dropna().reset_index(drop=True)
    calCI_df['DateString']= calCI_df['DateFormat'].dt.strftime('%Y-%m-%d')
    #print(calCI_df.head(10))
    print(calCI_df.shape)



    #read calendar economicActibity important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\economicActivity_calendar.txt'  
    cols = ['DateInput']
    calEC_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calEC_df['DateInput'] = calEC_df['DateInput'].str.strip()
    #print(calEC_df.head(10))
    calEC_df['DateFormat'] = pd.to_datetime(calEC_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calEC_df = calEC_df.dropna().reset_index(drop=True)
    calEC_df['DateString']= calEC_df['DateFormat'].dt.strftime('%Y-%m-%d')
    #print(calEC_df.head(10))
    print(calEC_df.shape)


    #read calendar employment_calendar important dates
    pathCalendarFile = r'E:\Work\GitHub\Python\employment_calendar.txt'  
    cols = ['DateInput']
    calEM_df = pd.read_csv(pathCalendarFile, header=None, names=cols, delimiter="@")
    calEM_df['DateInput'] = calEM_df['DateInput'].str.strip()
    #print(calEM_df.head(10))
    calEM_df['DateFormat'] = pd.to_datetime(calEM_df['DateInput'], format='%A, %B %d, %Y', errors='coerce')
    calEM_df = calEM_df.dropna().reset_index(drop=True)
    calEM_df['DateString']= calEM_df['DateFormat'].dt.strftime('%Y-%m-%d')
    #print(calEM_df.head(10))
    print(calEM_df.shape)



    my_df['Centralbank'] = np.where( np.isin(my_df['dateOnly'], calCB_df['DateString'].values) , 1, 0)
    my_df['ConfidenceIndex'] = np.where( np.isin(my_df['dateOnly'], calCI_df['DateString'].values) , 1, 0)
    my_df['EconomicActivity'] = np.where( np.isin(my_df['dateOnly'], calEC_df['DateString'].values) , 1, 0)
    my_df['Employment'] = np.where( np.isin(my_df['dateOnly'], calEM_df['DateString'].values) , 1, 0)
    print(my_df)



    my_df = my_df.reset_index() 
    my_df = my_df.dropna().reset_index(drop=True)

    #print(my_df.head(10))

    print("Finish reading files to dataframe from ", path)
    return my_df




def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    global stockList
    


    dfTestFeat = dfTest[colsFeat]

    forecasts = model.predict_proba(dfTestFeat)


    #result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecastSell", "forecastBuy"]) ], axis=1)
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecastSell", "not_forecast", "forecastBuy"]) ], axis=1)

    result['forecast'] = np.where(result['forecastSell'] > result['forecastBuy'], result['forecastSell'], result['forecastBuy'])
    result['action'] = np.where(result['forecastSell'] > result['forecastBuy'], 'sell', 'buy')
    result['pnlAbs'] = abs(result['PNL'])


    
    for k in range(len(result)):
        if(result["forecastSell"].iloc[k]> result["forecastBuy"].iloc[k]):
            result["PNL"].iloc[k] = - result["PNL"].iloc[k]


    time_df = result.groupby(["Date"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
    time_df[ "PNLPerTrade"] =  time_df["PNL"] / time_df["Trades"]
    #time_df.to_csv('E:\Work\GitHub\Energy\output/my_dfdatetime.txt', sep=';', index=True, mode='a' )

    totalBuy = 0
    meanBuy = 0
    noGain = 0
    noLost = 0
    noBuy = 0
    noLong = 0
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

        for p in range(1):
            totalBuy = totalBuy + timeResult["PNL"].iloc[p]
            if(timeResult["forecastSell"].iloc[p] < timeResult["forecastBuy"].iloc[p]):
                noLong = noLong +1

            if(timeResult["PNL"].iloc[p] > 0):
                noGain = noGain +1
            else:
                noLost = noLost +1
            noBuy = noBuy +1

    meanBuy = totalBuy /noBuy
    print(str(no)+". TotalBuy="+str( round(totalBuy,2) )+" meanBuy="+str(meanBuy)+" acuracy="+str(noGain/noBuy)+" acuracyGlobal="+str(noGainGlobal/(noGainGlobal+noLostGlobal))+" long="+str(noLong/noBuy))
    totalPNL = totalPNL +totalBuy






# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================



#print("Before reversal Array is :",stockList)
 
#stockList.reverse() #reversing using reverse()
#print("After reversing Array:",stockList)



#Read data
dfData=read_df(pathGlobal, indexList)

#dfFit1 = dfData[(dfData['Date'] >= "2021-02-01") & (dfData['Date'] < "2023-01-01") ]
dfFit1 = dfData[ (dfData['Date'] < "2024-01-01") ]

dfTest1 = dfData[(dfData['Date'] >= "2024-01-01") & (dfData['Date'] < "2024-02-01") ]
dfTest2 = dfData[(dfData['Date'] >= "2024-02-01") & (dfData['Date'] < "2024-03-01") ]
dfTest3 = dfData[(dfData['Date'] >= "2024-03-01") & (dfData['Date'] < "2024-04-01") ]
dfTest4 = dfData[(dfData['Date'] >= "2024-04-01") & (dfData['Date'] < "2024-05-01") ]
dfTest5 = dfData[(dfData['Date'] >= "2024-05-01") & (dfData['Date'] < "2024-06-01") ]
dfTest6 = dfData[(dfData['Date'] >= "2024-06-01") & (dfData['Date'] < "2024-07-01") ]





colsFeat = []

for i in range(0, pretunnelLenght):
    colsFeat.append('Open_'+str(i))
    colsFeat.append('High_'+str(i))
    colsFeat.append('Low_'+str(i))
    colsFeat.append('Close_'+str(i))
    colsFeat.append('Volume_'+str(i))
for x in indexList:
    for i in range(0, pretunnelLenght):
        colsFeat.append('Open_'+x+'_'+str(i))
        colsFeat.append('High_'+x+'_'+str(i))
        colsFeat.append('Low_'+x+'_'+str(i))
        colsFeat.append('Close_'+x+'_'+str(i))
        colsFeat.append('Volume_'+x+'_'+str(i))

colsFeat.append('Hour')
colsFeat.append('Minute')
colsFeat.append('DayOfWeek')
colsFeat.append('DayOfMonth')
colsFeat.append('Month')

#colsFeat.append('Centralbank')
#colsFeat.append('ConfidenceIndex')
#colsFeat.append('EconomicActivity')
#colsFeat.append('Employment')


 

dfFit1 = dfFit1.sort_values(by=['Date'], ascending=True)
dfFit1 = dfFit1.reset_index() 


dfTest1 = dfTest1.sort_values(by=['Date'], ascending=True)
dfTest2 = dfTest2.sort_values(by=['Date'], ascending=True)
dfTest3 = dfTest3.sort_values(by=['Date'], ascending=True)
dfTest4 = dfTest4.sort_values(by=['Date'], ascending=True)
dfTest5 = dfTest5.sort_values(by=['Date'], ascending=True)
dfTest6 = dfTest6.sort_values(by=['Date'], ascending=True)

dfTest1 = dfTest1.reset_index() 
dfTest2 = dfTest2.reset_index() 
dfTest3 = dfTest3.reset_index()
dfTest4 = dfTest4.reset_index()
dfTest5 = dfTest5.reset_index()
dfTest6 = dfTest6.reset_index() 



features = dfFit1[colsFeat]
target = dfFit1['target']

print(dfFit1)
print(dfTest1)

''' 
print(dfFit1[[ 'Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfFit1['target'].value_counts())

print(dfTest1[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfTest2[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfTest3[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfTest4[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfTest5[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
print(dfTest6[['Open_0','Open_1','Open_5','Open_9','High_9','Low_9','Close_9','Open_SPY_9','PNL']].describe())
'''

print(dfFit1['target'].value_counts())


scaler = StandardScaler()
#features = scaler.fit_transform(features)




X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=1, shuffle=False )
print("X= ", X_train.shape, X_valid.shape)
print("Y= ", Y_train.shape, Y_valid.shape)




modelsName = ["CatBoostClassifier", "CatBoostClassifier_2", "CatBoostClassifier_3", "CatBoostClassifier_4", "XGBClassifier", "XGBClassifier_2", "XGBClassifier_3", "XGBClassifier_3", "LGBMClassifier" , "LGBMClassifier_2", "LGBMClassifier_3", "LGBMClassifier_4",
              "LogisticRegression", "LogisticRegression2", "LogisticRegression3", "LogisticRegression4", "LogisticRegressionPipe1", "LogisticRegressionPipe2", "LogisticRegressionPipe3",
               "LinearDiscriminantAnalysis", "LinearDiscriminantAnalysis2", "LinearDiscriminantAnalysis3",   "LinearDiscriminantAnalysisPipe1", "LinearDiscriminantAnalysisPipe2", "LinearDiscriminantAnalysisPipe3",
               "QuadraticDiscriminantAnalysis", "HistGradientBoostingClassifier",
              "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "ExtraTreesClassifier2", "ExtraTreesClassifier3", "ExtraTreesClassifier4", 
              "KNeighborsClassifier", "GaussianNB", "GradientBoostingClassifier", "BaggingClassifier", "AdaBoostClassifier", 
              "MLPClassifier", "MLPClassifier2", "XGBClassifier1", "XGBClassifier2"]




models = [  CatBoostClassifier(verbose=0),
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", CatBoostClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform') ), ("clf", CatBoostClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer() ), ("clf", CatBoostClassifier(verbose=0) )]),
            XGBClassifier(verbose=0),
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", XGBClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform') ), ("clf", XGBClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer() ), ("clf", XGBClassifier(verbose=0) )]),
            LGBMClassifier(verbose=0), 
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", LGBMClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform') ), ("clf", LGBMClassifier(verbose=0) )]),
            Pipeline(steps=[("trans", QuantileTransformer() ), ("clf", LGBMClassifier(verbose=0) )]),
            LogisticRegression(),
            LogisticRegression(solver='liblinear'),
            LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.1),
            LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01),
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5) )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform') ), ("clf", LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5) )]),
            Pipeline(steps=[("trans", QuantileTransformer() ), ("clf", LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5) )]),
            LinearDiscriminantAnalysis(solver='svd'),
            LinearDiscriminantAnalysis(solver='lsqr'),
            LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr'),
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", LinearDiscriminantAnalysis() )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform') ), ("clf", LinearDiscriminantAnalysis() )]),
            Pipeline(steps=[("trans", QuantileTransformer() ), ("clf", LinearDiscriminantAnalysis() )]),
            QuadraticDiscriminantAnalysis(),
            HistGradientBoostingClassifier(),
            DecisionTreeClassifier(), 
            RandomForestClassifier(verbose=0), 
            ExtraTreesClassifier(verbose=0),
            Pipeline(steps=[("trans", RobustScaler(quantile_range=(25, 75)) ), ("clf", ExtraTreesClassifier() )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='uniform')  ), ("clf", ExtraTreesClassifier() )]),
            Pipeline(steps=[("trans", QuantileTransformer(output_distribution='normal') ), ("clf", ExtraTreesClassifier() )]),
            neighbors.KNeighborsClassifier(),
            GaussianNB(),   
            GradientBoostingClassifier(),
            BaggingClassifier( ),
            AdaBoostClassifier( ),
            MLPClassifier(activation='relu', max_iter=20000, hidden_layer_sizes=(150,100,150), learning_rate_init=0.01, random_state=0),
            MLPClassifier(activation='identity', max_iter=100000, hidden_layer_sizes=(200,150,100,150), learning_rate_init=0.001, random_state=0),
            XGBClassifier(predictor='cpu_predictor',max_depth=6, learning_rate=0.001),
            XGBClassifier(predictor='gpu_predictor',max_depth=6, learning_rate=0.001)
        ]
 
for i in range(0, 42):
    print('clungu start training no ' +str(i)+' for model '+str(models[i])+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))

    models[i].fit(X_train, Y_train)





    print('Training Accuracy : ', models[i].score(X_train, Y_train) )
    print('Validation Accuracy : ', models[i].score(X_valid, Y_valid)  )
    #print('Log Lost : ', metrics.log_loss(Y_valid, models[i].predict(X_test_trans)))
    print('F1 Score : ', f1_score(Y_valid, models[i].predict(X_valid), average=None))

    
    

    totalPNL=0

    checkForecast(dfTest1, models[i], 1)
    checkForecast(dfTest2, models[i], 2)
    checkForecast(dfTest3, models[i], 3)
    checkForecast(dfTest4, models[i], 4)
    checkForecast(dfTest5, models[i], 5)
    checkForecast(dfTest6, models[i], 6)



    if maxPNL < totalPNL:
        maxPNL = totalPNL

    print("==> TotalPNL="+str(totalPNL)+" while current maxPNL="+str(maxPNL))
    print("----------------------------------------------------------------------")

    

    # save the model to disk
    #filename = str(modelsName[i])+ '_model.sav'
    #pickle.dump(models[i], open(filename, 'wb'))

    
    print("Finish model", modelsName[i] +' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------------------------------")