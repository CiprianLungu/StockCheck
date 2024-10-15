import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import catboost as cb
import time
import pickle 
import warnings

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
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, log_loss


from pathlib import Path
from os import path


warnings.filterwarnings('ignore')

#Global settings
format=10000
pretunnelLenght=10
tunnelLenght=1

tradeperday=10

maxPNL=0
totalPNL=0
totalTrades=0


pathGlobal = r'E:\common\stock\data_DAY'  
pathTest1 = r'E:\common\stock\stock_data\data_202401' 
pathTest2 = r'E:\common\stock\stock_data\data_202402' 

pathTest1b = r'E:\common\stock\stock_data\oldAlpaca\data_202309'
pathTest2b = r'E:\common\stock\stock_data\oldAlpaca\data_202310'
pathTest3b = r'E:\common\stock\stock_data\oldAlpaca\data_202311'
pathTest4b = r'E:\common\stock\stock_data\oldAlpaca\data_202312'

''' 
#stockList=  [ 'TSLA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'EEM', 'INTC', 'KVUE', 'SLV', 'DIS', 'BABA', 'TLT', 'PLTR', 'GOOGL', 'BAC', 'NFLX', 'C', 'XLF', 'WBD', 'ROKU', 'GOOG', 'PYPL', 'PFE', 'F', 'SQ', 'JNJ', 'XOM', 'SOXL', 'COIN', 'AFRM', 'OXY', 'EWZ', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SNAP', 'SHOP', 'QCOM', 'JPM', 'PARA', 'WBA', 'T', 'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'AVGO', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'ENPH', 'CVX', 'VALE', 'DG', 'DIA', 'CHWY', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'PINS', 'CRWD', 'VZ', 'KO', 'WMT', 'ET', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'LYFT', 'GTLB', 'CVS', 'MRNA', 'U', 'FSLR', 'ETSY', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'HOOD', 'GS', 'LUV', 'MS', 'PTON', 'SBUX', 'ADBE', 'CCJ', 'DAL', 'RH', 'PEP', 'MDB', 'RIG', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'AEO', 'COST', 'UNH', 'NU']
stockList=  ['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'SLV', 'DIS', 
             'BABA', 'TLT', 'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'JNJ', 'XOM', 'COIN', 'AFRM', 'OXY', 'CVNA', 'ORCL', 'DKNG', 'SHOP', 'JPM', 'T', 
             'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'SE', 'CVX', 'VALE', 'DG', 'NEE', 'FCX', 'BX', 'CRM', 'CRWD', 
             'VZ', 'KO', 'UPS', 'DVN', 'UAL', 'DELL', 'SMCI', 'FSLR', 'LLY', 'KR', 'NEM', 'GS', 'MS', 'SBUX', 'ADBE', 
             'PEP', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'OKTA', 'COST', 'UNH',
             'DELL', 'PYPL', 'RIOT', 'SNAP', 'AEO', 'EEM', 'C', 'CCJ', 'KVUE', 'DAL', 'PLTR', 'HOOD', 'MARA', 'PINS']


stockList=  [ 'SPY', 'DIA', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLV', 'NVDA', 'AMD', 'MSFT', 'DIS', 
              'GOOGL', 'ROKU', 'GOOG', 'SQ', 'XOM', 'COIN', 'AFRM', 'OXY', 'GM', 'CVNA', 'DKNG', 'SHOP', 'JPM', 
             'T', 'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'WFC', 'EFA', 'TGT', 'CVX', 'VALE', 'BX', 
             'CRM', 'CRWD', 'UPS', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'CLF', 'GS', 'MS', 
             'TMUS', 'MRVL', 'X', 'AEO']

#all
stockList=  ['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'MRNA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'SLV', 'DIS', 
             'BABA', 'TLT', 'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'JNJ', 'XOM', 'COIN', 'AFRM', 'OXY', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SHOP', 'JPM', 'WBA', 'T', 
             'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'CVX', 'VALE', 'DG', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'CRWD', 
             'VZ', 'KO', 'WMT', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'GS', 'LUV', 'MS', 'SBUX', 'ADBE', 
             'PEP', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'COST', 'UNH',
             'DELL', 'BAC', 'PYPL', 'RIOT', 'SNAP', 'PARA', 'AEO', 'PFE', 'EEM', 'CHWY', 'C', 'CCJ', 'F', 'INTC', 'WBD', 'KVUE', 'DAL', 'PLTR', 'ENPH', 'ETSY', 'HOOD', 'TSLA', 'RH', 'SOXL', 'LYFT', 'MARA', 'MDB', 'PINS', 'U']


'''

#all
stockList=  ['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'MRNA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'SLV', 'DIS', 
             'BABA', 'TLT', 'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'JNJ', 'XOM', 'COIN', 'AFRM', 'OXY', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SHOP', 'JPM', 'WBA', 'T', 
             'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'CVX', 'VALE', 'DG', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'CRWD', 
             'VZ', 'KO', 'WMT', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'GS', 'LUV', 'MS', 'SBUX', 'ADBE', 
             'PEP', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'COST', 'UNH',
             'DELL', 'BAC', 'PYPL', 'RIOT', 'SNAP', 'PARA', 'AEO', 'PFE', 'EEM', 'CHWY', 'C', 'CCJ', 'F', 'INTC', 'WBD', 'KVUE', 'DAL', 'PLTR', 'ENPH', 'ETSY', 'HOOD', 'TSLA', 'RH', 'SOXL', 'LYFT', 'MARA', 'MDB', 'PINS', 'U']



#indexList=['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
indexList=['SPY', 'DIA', 'IWM', 'QQQ']

#Read data from directory into a dataframe and transform
def read_df(pathD):
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
    print(my_df.head(3))

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)+'_orig']=my_df['Open'].shift(-i)
        my_df['High_'+str(i)+'_orig']=my_df['High'].shift(-i)
        my_df['Low_'+str(i)+'_orig']=my_df['Low'].shift(-i)
        my_df['Close_'+str(i)+'_orig']=my_df['Close'].shift(-i)
        my_df['Volume_'+str(i)+'_orig']=my_df['Volume'].shift(-i)
    my_df['Open_sale_orig']=my_df['Open'].shift(tunnelLenght+1)
    my_df['Close_sale_orig']=my_df['Close'].shift(tunnelLenght)
    my_df['Open_buy_orig']=my_df['Open'].shift(1)
    my_df['Date_sale_orig']=my_df['Date'].shift(tunnelLenght+1)


    my_df=my_df.dropna().reset_index(drop=True)

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)]=my_df['Open_'+str(i)+'_orig']*format//my_df['Open']
        my_df['High_'+str(i)]=my_df['High_'+str(i)+'_orig']*format//my_df['Open']
        my_df['Low_'+str(i)]=my_df['Low_'+str(i)+'_orig']*format//my_df['Open']
        my_df['Close_'+str(i)]=my_df['Close_'+str(i)+'_orig']*format//my_df['Open']
        my_df['Volume_'+str(i)]=my_df['Volume_'+str(i)+'_orig']*format//my_df['Volume']

        my_df = my_df.drop(['Open_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['High_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['Low_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['Close_'+str(i)+'_orig'], axis=1)
        my_df = my_df.drop(['Volume_'+str(i)+'_orig'], axis=1)


    print("after transform my_df.shape=", my_df.shape)
    #print(my_df.head(10))


    for x in range(len(indexList)):
        filePath = pathD+"\\"+indexList[x]+".txt"

        dia_df = pd.read_csv(filePath, header=None, names=cols)

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+str(i)+'_orig']=dia_df['Open'].shift(-i)
            dia_df['High_'+str(i)+'_orig']=dia_df['High'].shift(-i)
            dia_df['Low_'+str(i)+'_orig']=dia_df['Low'].shift(-i)
            dia_df['Close_'+str(i)+'_orig']=dia_df['Close'].shift(-i)
            dia_df['Volume_'+str(i)+'_orig']=dia_df['Volume'].shift(-i)

        #print("after raw read for stock %s has index_df.shape=",indexList[x], dia_df.shape)
        #print(dia_df.head(10))
        dia_df=dia_df.dropna().reset_index(drop=True)

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexList[x]+'_'+str(i)]=dia_df['Open_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['High_'+indexList[x]+'_'+str(i)]=dia_df['High_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Low_'+indexList[x]+'_'+str(i)]=dia_df['Low_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Close_'+indexList[x]+'_'+str(i)]=dia_df['Close_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Volume_'+indexList[x]+'_'+str(i)]=dia_df['Volume_'+str(i)+'_orig']*format//dia_df['Volume']

            dia_df = dia_df.drop(['Open_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['High_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Low_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Close_'+str(i)+'_orig'], axis=1)
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
    print(my_df.head(3))





    my_df['PNL_orig']=my_df['Open_sale_orig']-my_df['Open_buy_orig']
    #my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    #my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Open_buy_orig']
    
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    #my_df['target'] = np.where(my_df['PNL'] > 5, 2, np.where(my_df['PNL'] < -5, 0, 1))
    my_df['target'] = np.where(my_df['PNL'] < 0, 0, 1)

    #my_df['Volume_0']=100
    #for i in range(1, pretunnelLenght):
    #    my_df['Volume_'+str(i)] = np.where(my_df['Volume_'+str(i)+'_orig'] > my_df['Volume_'+str(i-1)+'_orig'], my_df['Volume_'+str(i-1)]+1,my_df['Volume_'+str(i-1)]-1 )





    my_df[['dateOnly', 'time']] = my_df['Date'].str.split('T', n=1, expand=True)
    #print(my_df.head(10))
    my_df[['Hour', 'minute']] = my_df['time'].str.split(':', n=1, expand=True)
    my_df[['year', 'Month', 'day']] = my_df['dateOnly'].str.split('-', n=2, expand=True)
    #print(my_df.head(10))

    my_df['DateFormat'] = pd.to_datetime(my_df['Date'], format='ISO8601')
    my_df['DayOfMonth']= my_df['DateFormat'].apply(lambda x: x.day)
    my_df['DayOfWeek']= my_df['DateFormat'].apply(lambda x: x.weekday())
    my_df['Month']= my_df['DateFormat'].apply(lambda x: x.month)
    my_df['Hour']= my_df['DateFormat'].apply(lambda x: x.hour)






    

    #my_df = my_df.drop(["Date"], axis=1)
    my_df = my_df.drop(["Open"], axis=1)
    my_df = my_df.drop(["High"], axis=1)
    my_df = my_df.drop(["Low"], axis=1)
    my_df = my_df.drop(["Close"], axis=1)
    my_df = my_df.drop(["Open_buy_orig"], axis=1)
    my_df = my_df.drop(["Open_sale_orig"], axis=1)
    my_df = my_df.drop(["minute"], axis=1)
    my_df = my_df.drop(["year"], axis=1)





    my_df = my_df.reset_index() 

    print(my_df.head(3))

    print("Finish reading files to dataframe from ", path)
    return my_df



def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    global totalTrades
    dfTestFeat = dfTest[colsFeat].copy()
    #forecasts = model.predict(dfTestFeat)
    forecasts = model.predict_proba(dfTestFeat)

    #result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecast"]) ], axis=1)
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=[ "forecastSell", "forecastBuy"]) ], axis=1)
    result['forecast'] = np.where(result['forecastSell'] > 0.5, result['forecastSell'], result['forecastBuy'])
    result['action'] = np.where(result['forecastSell'] > 0.5, 'sell', 'buy')
    result['pnlAbs'] = abs(result['PNL'])



    maxDayOfMonth = result['DayOfMonth'].max()
    #print('maxDayOfMonth='+str(maxDayOfMonth))

     

    monthPnl = 0
    noMonth = 0

    noGain = 0
    noLost = 0
    

    if(no!=0):
        for k in range(len(result)):            
            if(result["forecastSell"].iloc[k] > 0.5):
                result["PNL"].iloc[k] = - result["PNL"].iloc[k]


        for dm in range(1, maxDayOfMonth+1):

            day_df = result.loc[(result['DayOfMonth'] == dm)]
            if(len(day_df)==0):
                continue

            day_df = day_df.sort_values(by=['forecast'], ascending=False)
            dayPNL = 0

            trade_df = day_df.head(tradeperday)
            
            for index, row in trade_df.iterrows():
                
                if(row["forecastBuy"] > 0.5):
                    dayPNL = dayPNL + row["PNL"]
                    noMonth = noMonth +1
                elif(row["forecastSell"] > 0.5):
                    dayPNL = dayPNL + row["PNL"]
                    noMonth = noMonth +1
                    
                if(row["PNL"] > 0):
                    noGain = noGain +1
                elif(row["PNL"] < 0):
                    noLost = noLost +1

            monthPnl= monthPnl +dayPNL
            

    else:
        for k in range(len(result)):
                
            if(result["forecastSell"].iloc[k] > 0.5):
                result["PNL"].iloc[k] = - result["PNL"].iloc[k]
                monthPnl = monthPnl + result["PNL"].iloc[k]
                noMonth = noMonth +1
            elif(result["forecastBuy"].iloc[k] > 0.5):
                monthPnl = monthPnl + result["PNL"].iloc[k]
                noMonth = noMonth +1

            if(result["PNL"].iloc[k] > 0):
                noGain = noGain +1
            elif(result["PNL"].iloc[k] < 0):
                noLost = noLost +1


    totalTrades = totalTrades + noMonth

    my_dfG = result.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
    my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
    my_dfG = my_dfG.sort_values(by=['PNLPerTrade'], ascending=True)
    #print(my_dfG)
    my_dfG.to_csv('E:\Work\GitHub\Energy\output/my_dfG.txt', sep=';', index=True, mode='a' )

    
     
    
    



    

    #print("-----------------------------------")
    print(str(no)+". monthPnl="+str(monthPnl)+" noMonth="+str(noMonth)+" mean="+str(monthPnl/noMonth)+" --> noGain="+str(noGain)+" noLost="+str(noLost)+" Accuracy="+str(noGain/noMonth))
    totalPNL = totalPNL +monthPnl




# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================
 

#Read data
dfData=read_df(pathGlobal)
print(dfData.head())


print ('len(stockList)=',len(stockList))
time.sleep(3)


dfFit1 = dfData[(dfData['Date'] >= "2021-07-01") & (dfData['Date'] < "2023-07-01") ]
dfTest1 = dfData[(dfData['Date'] >= "2023-07-01") & (dfData['Date'] < "2023-08-01") ]
dfTest2 = dfData[(dfData['Date'] >= "2023-08-01") & (dfData['Date'] < "2023-09-01") ]
dfTest3 = dfData[(dfData['Date'] >= "2023-09-01") & (dfData['Date'] < "2023-10-01") ]
dfTest4 = dfData[(dfData['Date'] >= "2023-10-01") & (dfData['Date'] < "2023-11-01") ]
dfTest5 = dfData[(dfData['Date'] >= "2023-11-01") & (dfData['Date'] < "2023-12-01") ]
dfTest6 = dfData[(dfData['Date'] >= "2023-12-01") & (dfData['Date'] < "2024-01-01") ]


dfTestAll = dfData[(dfData['Date'] >= "2023-07-01") & (dfData['Date'] < "2024-01-01") ]

dfFit1 = dfFit1.reset_index() 
dfTest1 = dfTest1.reset_index() 
dfTest2 = dfTest2.reset_index() 
dfTest3 = dfTest3.reset_index() 
dfTest4 = dfTest4.reset_index() 
dfTest5 = dfTest5.reset_index() 
dfTest6 = dfTest6.reset_index() 

dfTestAll = dfTestAll.reset_index() 


    





colsFeat = []
colsFeat.append('Hour')
#colsFeat.append('Month')
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


features = dfFit1[colsFeat]
target = dfFit1['target']


#scaler = StandardScaler()
#features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.2, random_state=104, shuffle=True )
print("X= ", X_train.shape, X_valid.shape)
print("Y= ", Y_train.shape, Y_valid.shape)




modelsName = ["CatBoostClassifier_1", "CatBoostClassifier_2", "CatBoostClassifier_3",  "CatBoostClassifier_4",  "CatBoostClassifier_5",
              "XGBClassifier_1", "XGBClassifier_2", "XGBClassifier_3", 
               "LGBMClassifier_1", "LGBMClassifier_2", "LGBMClassifier_3",  
               "ExtraTreesClassifier_1",  "ExtraTreesClassifier_2" ]
              

models = [ CatBoostClassifier( iterations=2000),
        CatBoostClassifier(verbose=0, bagging_temperature=0.26, depth=2, l2_leaf_reg=61, learning_rate=0.0061,random_strength=0.05336 ), 
        CatBoostClassifier( iterations=2000, learning_rate=0.03, depth=10, random_seed=127, l2_leaf_reg=61), 
        CatBoostClassifier( iterations=10000, learning_rate=0.001, depth=10, random_seed=127, od_pval=100), 
        CatBoostClassifier( iterations=5914, learning_rate=0.00348320476177015, depth=8, random_seed=127, od_pval=100), 
        XGBClassifier(verbose=0), 
        XGBClassifier(verbose=0, subsample=0.26, max_depth =12, n_estimators=1000, learning_rate=0.003), #best,
        XGBClassifier(verbose=0, subsample=0.26, max_depth =12, n_estimators=1000, learning_rate=0.009), # good
        LGBMClassifier(verbose=0), 
        LGBMClassifier(verbose=-1, learning_rate=0.019244449765581838, boosting_type='dart', n_estimators=10000, min_data=73, max_depth=64), 
        LGBMClassifier(verbose=-1, learning_rate=0.000221091301252091, boosting_type='goss', n_estimators=20000, min_data=59, max_depth=46), 
        ExtraTreesClassifier(), 
        ExtraTreesClassifier(max_depth=25, n_estimators=6843, min_samples_split=3, min_samples_leaf=4), 
        ExtraTreesClassifier(max_depth=49, n_estimators=1273, min_samples_split=8, min_samples_leaf=5) 
]

for i in range(8, 12):
    print('----------------------------------------------------------------------------------------------')
    print('Start training no ' +str(i)+' for model '+str(modelsName[i]) +' at time  '+datetime.datetime.now().strftime("%H:%M:%S") )




    if "CatBoostClassifier" in str(modelsName[i]):
        models[i].fit(X_train.values, Y_train, eval_set=(X_valid, Y_valid), plot=True, early_stopping_rounds=100)
    elif "XGBClassifier" in str(modelsName[i]):
        eval_set1 = [(X_train, Y_train), (X_valid, Y_valid)]
        models[i].fit(X_train, Y_train, eval_metric=["mlogloss", "merror"], eval_set=eval_set1, verbose=False)
    elif "LGBMClassifier" in str(modelsName[i]):
        models[i].fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)], eval_metric=['multi_logloss', 'multi_error'])
    elif "ExtraTreesClassifier" in str(modelsName[i]):
        models[i].fit(X_train, Y_train)
    
    print('Validation Score : ', models[i].score(X_valid.values, Y_valid) )



    Y_pred = models[i].predict(X_valid)
    accuracy = accuracy_score(Y_valid, Y_pred)
    f1 = f1_score(Y_valid, Y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")

    totalPNL=0       

    checkForecast(dfTestAll, models[i], 0)

    checkForecast(dfTest1, models[i], 1)
    checkForecast(dfTest2, models[i], 2)
    checkForecast(dfTest3, models[i], 3)
    checkForecast(dfTest4, models[i], 4)
    checkForecast(dfTest5, models[i], 5)
    checkForecast(dfTest6, models[i], 6)


    

    if maxPNL < totalPNL:
        maxPNL = totalPNL

    print("==> TotalPNL="+str(totalPNL)+" while current maxPNL="+str(maxPNL) )
    print("----------------------------------------------------------------------")
    print("Finish model"+str(modelsName[i]) +' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
    print("----------------------------------------------------------------------")


    if "CatBoostClassifier" in str(modelsName[i]):
        eval_results = models[i].get_evals_result()

        # Extract the metrics
        train_loss = eval_results['learn']['Logloss']
        val_loss = eval_results['validation']['Logloss']
        #train_accuracy = [accuracy_score(Y_train, np.argmax(pred, axis=1)) for pred in models[i].predict_proba(X_train)]
        #val_accuracy = [accuracy_score(Y_valid, np.argmax(pred, axis=1)) for pred in models[i].predict_proba(X_valid)]

        # Plot training and validation loss
        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        ax[0].plot(epochs, train_loss, label='Train Loss')
        ax[0].plot(epochs, val_loss, label='Validation Loss')
        ax[0].set_title('CatBoost Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Plot training and validation accuracy
        #ax[1].plot(epochs, train_accuracy, label='Train Accuracy')
        #ax[1].plot(epochs, val_accuracy, label='Validation Accuracy')
        #ax[1].set_title('CatBoost Accuracy')
        #ax[1].set_xlabel('Epochs')
        #ax[1].set_ylabel('Accuracy')
        #ax[1].legend()

        ax[1].plot(epochs, 1 - np.array(eval_results['learn']['Logloss']), label='Train Accuracy')
        ax[1].plot(epochs, 1 - np.array(eval_results['validation']['Logloss']), label='Test Accuracy')
        ax[1].set_title(str(modelsName[i])+' Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.show()

    elif "XGBClassifier" in str(modelsName[i]):

        eval_results = models[i].evals_result()

        # Extract the metrics
        epochs = len(eval_results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)

        # Plot training and validation loss
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        ax[0].plot(x_axis, eval_results['validation_0']['mlogloss'], label='Train Loss')
        ax[0].plot(x_axis, eval_results['validation_1']['mlogloss'], label='Validation Loss')
        ax[0].set_title('XGBoost Log Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Log Loss')
        ax[0].legend()

        # Plot training and validation error
        ax[1].plot(x_axis, eval_results['validation_0']['merror'], label='Train Error')
        ax[1].plot(x_axis, eval_results['validation_1']['merror'], label='Validation Error')
        ax[1].set_title('XGBoost Classification Error')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Classification Error')
        ax[1].legend()

        #plt.show()
    elif "LGBMClassifier" in str(modelsName[i]):
        # Retrieve the logged metrics
        eval_results = models[i].evals_result_


        # Extract the metrics
        epochs = len(eval_results['training']['binary_logloss'])
        x_axis = range(0, epochs)

        # Plot training and validation loss
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        ax[0].plot(x_axis, eval_results['training']['binary_logloss'], label='Train Loss')
        ax[0].plot(x_axis, eval_results['valid_1']['binary_logloss'], label='Validation Loss')
        ax[0].set_title('LGBM Log Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Log Loss')
        ax[0].legend()

        # Plot training and validation error
        ax[1].plot(x_axis, eval_results['training']['binary_error'], label='Train Error')
        ax[1].plot(x_axis, eval_results['valid_1']['binary_error'], label='Validation Error')
        ax[1].set_title('LGBM Classification Error')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Classification Error')
        ax[1].legend()

        plt.show()
    elif "ExtraTreesClassifier" in str(modelsName[i]):

        # Predict probabilities and classes for the training and test set
        y_train_pred = models[i].predict(X_train)
        y_test_pred = models[i].predict(X_valid)
        y_train_pred_proba = models[i].predict_proba(X_train)
        y_test_pred_proba = models[i].predict_proba(X_valid)

        # Calculate accuracy and log loss
        train_accuracy = accuracy_score(Y_train, y_train_pred)
        test_accuracy = accuracy_score(Y_valid, y_test_pred)
        train_log_loss = log_loss(Y_train, y_train_pred_proba)
        test_log_loss = log_loss(Y_valid, y_test_pred_proba)

        # Collect accuracy and log loss over epochs for plotting (since ExtraTrees doesn't have epochs, we use this way)
        epochs = range(1)
        train_accuracies = [train_accuracy]
        test_accuracies = [test_accuracy]
        train_log_losses = [train_log_loss]
        test_log_losses = [test_log_loss]

        # Plot training and test accuracy and log loss
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        ax[0].plot(epochs, train_log_losses, label='Train Log Loss')
        ax[0].plot(epochs, test_log_losses, label='Test Log Loss')
        ax[0].set_title('ExtraTrees Log Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Log Loss')
        ax[0].legend()

        ax[1].plot(epochs, train_accuracies, label='Train Accuracy')
        ax[1].plot(epochs, test_accuracies, label='Test Accuracy')
        ax[1].set_title('ExtraTrees Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        #plt.show()



    # save the model to disk
    #filename = "runModel\\FRImbalance_Oficial\\"+str(modelsName[i])+'_model.sav'
    #pickle.dump(models[i], open(filename, 'wb'))

print('--->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINISH APPLY ANALYSIS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


