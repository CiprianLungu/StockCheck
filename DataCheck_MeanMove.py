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
pretunnelLenght=6
tunnelLenght=12

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




stockList = [ "SPY", "IWM", "QQQ", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "EEM", "INTC", "KVUE", "SLV", "TLT", "PLTR", "BAC", "MARA",
        "C", "XLF", "WBD", "PYPL", "PFE", "F", "XOM", "SOXL", "EWZ", "GM", "AAL", "RIOT", "DKNG", "SNAP", "PARA", "WBA", "T", "CCL", "PBR",
        "SNOW", "NKE", "WFC", "EFA", "XLU", "SE", "VALE", "DG", "CHWY", "NEE", "FCX", "PINS", "WMT", "ET", "DVN", "UAL", "PATH", 
        "LYFT", "U", "LLY", "KR", "PENN", "CLF", "LUV", "PTON", "DAL", "RH", "RIG", "TMUS", "MRVL", "CMCSA", "W", "AEO", "UNH" ]




#indexList=[ 'SPY', 'IWM', 'QQQ']
indexList=['SPY', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


#Read data from directory into a dataframe and transform
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


    print("after transform my_df.shape=", my_df.shape)
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

    #my_df['MonthOfTrimestre'] = my_df['Month'] % 3
    #my_df['Trimestre'] = my_df['Month'] // 3

    
    my_df.loc[( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") ),   "Hour" ] = my_df['Hour'] -1


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

    my_df.drop(my_df[ (my_df["time"] > "20:30:00Z") & 
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

    my_df.drop(my_df[(my_df["time"] > "19:30:00Z") & 
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

    #print(my_df.head(10))

    print("Finish reading files to dataframe from ", path)
    return my_df



def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    global stockList

    colsFeat = []

    dfTestFeat = dfTest[colsFeat].copy()
    #forecasts = model.predict(dfTestFeat)
    forecasts = model.predict_proba(dfTestFeat)



    #result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecast"]) ], axis=1)
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecastSell", "forecastBuy"]) ], axis=1)
    #result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=[ "forecastSell", "forecastStay", "forecastBuy"]) ], axis=1)
    #print(result.head)
    result['forecast'] = np.where(result['forecastSell'] > 0.5, result['forecastSell'], result['forecastBuy'])

    result['action'] = np.where(result['forecastSell'] > 0.5, 'sell', 'buy')
    result['pnlAbs'] = abs(result['PNL'])

    #print("result.shape=", result.shape)
    buyResult = result.nlargest(2000, "forecast")

    if(no==0):
        buyResult = result.nlargest(2000, "forecast")


    #print(buyResult.head)
    #print(buyResult.head)

    totalBuy = 0
    meanBuy = 0
    noGain = 0
    noLost = 0
    for k in range(len(buyResult)):
        if(buyResult["forecastSell"].iloc[k]> 0.5):
            totalBuy = totalBuy - buyResult["PNL"].iloc[k]
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


    my_dfG = buyResult.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
    my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
    my_dfG = my_dfG.sort_values(by=['PNLPerTrade'], ascending=True)
    #print(my_dfG)
    #if(no==0):
        #my_dfG.to_csv('E:\Work\GitHub\Python\Python\output/my_dfG66_catB.txt', sep=';', index=True, mode='a' )
        #print(my_dfG.head(3))
        #print("stock to be removed:" + str(my_dfG["File"].iloc[0]) )
        #stockList.remove(str(my_dfG["File"].iloc[0]))
        #print("stock has been removed removed:" + str(my_dfG["File"].iloc[0]) )




    #print("-----------------------------------")
    print(str(no)+". TotalBuy="+str(totalBuy)+" meanBuy="+str(meanBuy)+" acuracy="+str(noGain/len(buyResult)))
    totalPNL = totalPNL +totalBuy

    return str(my_dfG["File"].iloc[0])






# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================




#Read data
dfData=read_df(pathGlobal, indexList)

dfData['pnlAbs'] = abs(dfData['PNL'])

my_dfG = dfData.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
my_dfG = my_dfG.sort_values(by=['pnlAbsM'], ascending=True)

print(my_dfG.to_string())

