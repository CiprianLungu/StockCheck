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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, neighbors

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import accuracy_score
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
minPNL =0
totalPNL=0
modelPNL=0

bestModelPNL = 0
bestMinPNL = -100000
bestModelDescription= ""
bestModelDescription2= ""
bestModelDescription3= ""
bestModelDescription4= ""
bestModelDescription5= ""


pathGlobal = r'E:\common\stock\data_all_5min'  



#"MARA", "RIOT", "SOXL",  "XLU" , "XLF"
stockList = [ "PTON", "W", "RIG", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH","LYFT", "PENN", "SNOW", "SE", "CLF",
            "XLE", "INTC", "KVUE", "BAC",
            "C", "WBD", "PYPL", "F", "XOM", "EWZ", "GM", "AAL", "PARA", "WBA", "CCL", "PBR",
            "WFC", "VALE", "DG", "NEE", "FCX", "PINS", "ET", "DVN", "UAL",
            "LLY", "LUV", "DAL", "MRVL", "AEO" ]

#     "EEM", "EFA", "TLT", "XLP", "XLV" "SPY", "XLI", "WMT", "XLF", "DIA", "XLU", "XLB", "T", "QQQ", "TMUS", "IWM", "XLK", "PFE", "UNH", "SLV", "CMCSA" , "KR", "NKE", 

#    "SOXL", "RIOT", "MARA", "RIG", "W", "PTON", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH", "CLF", "SE", "SNOW", "PENN","LYFT", 
# 
# 


indexList=[ 'SPY', 'IWM', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLP']
#indexList=['SPY', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


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

    '''  
    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)]=my_df['Open_'+str(i)] - format
        my_df['High_'+str(i)]=my_df['High_'+str(i)] - format
        my_df['Low_'+str(i)]=my_df['Low_'+str(i)] - format
        my_df['Close_'+str(i)]=my_df['Close_'+str(i)] - format
        my_df['Volume_'+str(i)]=my_df['Volume_'+str(i)] - format
    '''

    print("after transform my_df.shape=", my_df.shape)
    print(my_df.head(3))


    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    #my_df['target'] = np.where(my_df['PNL'] > 5, 2, np.where(my_df['PNL'] < -5, 0, 1))
    my_df['target'] = np.where(my_df['PNL'] > 0, 1, 0)


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

        '''
        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexListCheck[x]+'_'+str(i)]=dia_df['Open_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['High_'+indexListCheck[x]+'_'+str(i)]=dia_df['High_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Low_'+indexListCheck[x]+'_'+str(i)]=dia_df['Low_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Close_'+indexListCheck[x]+'_'+str(i)]=dia_df['Close_'+indexListCheck[x]+'_'+str(i)] - format
            dia_df['Volume_'+indexListCheck[x]+'_'+str(i)]=dia_df['Volume_'+indexListCheck[x]+'_'+str(i)] - format
        '''

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
    for k in range(len(time_df)):
        #print(time_df["Date"].iloc[k])
        timeResult = result[(result["Date"] == time_df["Date"].iloc[k])]
        timeResult = timeResult.sort_values(by=['forecast'], ascending=False)

        if(len(timeResult) < 3):
            continue

        for p in range(2):
            totalBuy = totalBuy + timeResult["PNL"].iloc[p]
            if(timeResult["PNL"].iloc[p] > 0):
                noGain = noGain +1
            else:
                noLost = noLost +1
            noBuy = noBuy +1

    meanBuy = totalBuy /noBuy
    print(str(no)+". TotalBuy="+str( round(totalBuy,2) )+" meanBuy="+str(meanBuy)+" acuracy="+str(noGain/noBuy))
    totalPNL = totalPNL +totalBuy


    '''
    #initial valuation
    buyResult = result.nlargest(2000, "forecast")


    totalBuy = 0
    meanBuy = 0
    noBuy = 0
    noGain = 0
    noLost = 0
    for k in range(len(buyResult)):
        totalBuy = totalBuy + buyResult["PNL"].iloc[k]
        if(buyResult["PNL"].iloc[k] > 0):
            noGain = noGain +1
        else:
            noLost = noLost +1
        noBuy = noBuy +1
    meanBuy = totalBuy /len(buyResult)
     
    for k in range(len(buyResult)):
        if(buyResult["forecastSell"].iloc[k]> 0.5):
            totalBuy = totalBuy + buyResult["PNL"].iloc[k]
            if(buyResult["PNL"].iloc[k] > 0):
                noLost = noLost +1
            else:
                noGain = noGain +1
            buyResult["PNL"].iloc[k] = - buyResult["PNL"].iloc[k]
        else:
            totalBuy = totalBuy + buyResult["PNL"].iloc[k]
            if(buyResult["PNL"].iloc[k] > 0):
                noGain = noGain +1
            else:
                noLost = noLost +1
        totalBuy = totalBuy +1
    meanBuy = totalBuy /len(buyResult)
    '''






# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================







dateList = [ "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", "2021-12-01", 
            "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01", 
            "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"]



#Read data
dfDataAll=read_df(pathGlobal, indexList)

for L in range(2, len(indexList)):
    for subset in itertools.combinations(indexList, L):
        print("Running L=", L)
        print(subset)
        indexListCheck = subset

        colsFeat = []
        #colsFeat.append('Month')
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
        colsFeat.append('Hour')
        colsFeat.append('Minute')
        colsFeat.append('Month')
        #colsFeat.append('Trimestre')
        #4
        colsData = colsFeat.copy()
        colsData.append('Date')
        colsData.append('PNL')
        colsData.append('target')
        colsData.append('File')
        dfData = dfDataAll[colsData]



        for col in colsFeat:
            dfData[col] = dfData[col].astype('int')

        
        


        modelPNL = 0
        maxPNL =0
        minPNL = 50000
        print("------------------ START --------------------------------")
        for k in range(18):

            
            #print("current len(stockList="+str(len(stockList)))


            dateStartTraning = dateList[k]
            dateEndTraning = dateList[k+12*2]
            dateStartTest = dateList[k+12*2]
            dateEndTest = dateList[k+12*2+1]

            print(" ->> Check mounth : Traning : "+str(dateStartTraning)+" --> "+str(dateEndTraning)+"  Testing : "+str(dateStartTest)+" --> "+str(dateEndTest))

            dfFit1 = dfData[(dfData['Date'] >= dateStartTraning) & (dfData['Date'] < dateEndTraning) ]
            dfTest1 = dfData[(dfData['Date'] >= dateStartTest) & (dfData['Date'] < dateEndTest) ]

            dfFit1 = dfFit1.reset_index() 
            dfTest1 = dfTest1.reset_index() 


            features = dfFit1[colsFeat]
            target = dfFit1['target']


            #scaler = StandardScaler()
            #features = scaler.fit_transform(features)
            
            X_train, X_valid, Y_train, Y_valid = train_test_split(
                features, target, test_size=0.1, random_state=1, shuffle=False )


            model =  LGBMClassifier(verbose=0)


            model.fit(X_train, Y_train)
            
            print('Train Acc: : '+str(metrics.roc_auc_score(Y_train, model.predict(X_train)))+'  Valid Acc : '+str(metrics.roc_auc_score(Y_valid, model.predict(X_valid))))
            #print('Log Lost : ', metrics.log_loss(Y_valid, model.predict(X_valid)))    
            #print('Validation Score : ', model.score(X_valid, Y_valid) )


            totalPNL=0

            checkForecast(dfTest1, model, 0)


            if maxPNL < totalPNL:
                maxPNL = totalPNL
            if minPNL > totalPNL:
                minPNL = totalPNL    
            modelPNL = modelPNL + totalPNL

            

            print("==> TotalModelPNL="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL))
            #print("----------------------------------------------------------------------")

            del dfFit1
            del dfTest1


            gc.collect()

        if bestModelPNL < modelPNL:
            bestModelPNL = modelPNL
            bestModelDescription5 = bestModelDescription4
            bestModelDescription4 = bestModelDescription3
            bestModelDescription3 = bestModelDescription2
            bestModelDescription2 = bestModelDescription
            bestModelDescription= "Best model="+str("LGB")+" pnl="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL)+" with indexListCheck="+str(indexListCheck)
        elif bestMinPNL < minPNL:
            bestMinPNL = minPNL
            bestModelDescription5 = bestModelDescription4
            bestModelDescription4 = bestModelDescription3
            bestModelDescription3 = bestModelDescription2
            bestModelDescription2 = bestModelDescription
            bestModelDescription= "Best model="+str("LGB")+" pnl="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL)+" with indexListCheck="+str(indexListCheck)

        

        print("==> Total MODEL PNL="+str(modelPNL)+" while current maxPNL="+str(maxPNL)+" and current minPNL="+str(minPNL))
        print("----------------------------------------------------------------------")    
        print("bestModelDescription: "+str(bestModelDescription) )
        print("bestModelDescription2: "+str(bestModelDescription2) )
        print("bestModelDescription3: "+str(bestModelDescription3) )
        print("bestModelDescription4: "+str(bestModelDescription4) )
        print("bestModelDescription5: "+str(bestModelDescription5) )
        print("----------------------------------------------------------------------")

