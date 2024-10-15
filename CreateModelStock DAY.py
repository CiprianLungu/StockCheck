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


from pathlib import Path
from os import path


warnings.filterwarnings('ignore')

#Global settings
format=1000
pretunnelLenght=5
tunnelLenght=1

tradeperday=1000

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

stockList=  ['SPY', 'DIA', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLV', 'NVDA', 'AMD', 'MSFT', 'DIS', 
             'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'XOM', 'COIN', 'AFRM', 'OXY', 'GM', 'CVNA', 'DKNG', 'SHOP', 'JPM', 'T', 
             'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'WFC', 'EFA', 'TGT', 'SCHW', 'CVX', 'VALE', 'FCX', 'BX', 'CRM', 'CRWD', 
             'VZ', 'UPS', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'CLF', 'GS', 'MS', 
             'TMUS', 'MRVL', 'X', 'OKTA', 'COST', 'UNH',
             'DELL', 'BAC', 'AEO', 'C', 'CCJ', 'INTC', 'HOOD', 'LYFT', 'PINS']
'''

#all
stockList=  ['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'MRNA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'SLV', 'DIS', 
             'BABA', 'TLT', 'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'JNJ', 'XOM', 'COIN', 'AFRM', 'OXY', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SHOP', 'JPM', 'WBA', 'T', 
             'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'CVX', 'VALE', 'DG', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'CRWD', 
             'VZ', 'KO', 'WMT', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'GS', 'LUV', 'MS', 'SBUX', 'ADBE', 
             'PEP', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'COST', 'UNH',
             'DELL', 'BAC', 'PYPL', 'RIOT', 'SNAP', 'PARA', 'AEO', 'PFE', 'EEM', 'CHWY', 'C', 'CCJ', 'F', 'INTC', 'WBD', 'KVUE', 'DAL', 'PLTR', 'ENPH', 'ETSY', 'HOOD', 'TSLA', 'RH', 'SOXL', 'LYFT', 'MARA', 'MDB', 'PINS', 'U']




indexList=['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


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
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=[ "forecastBuy", "forecastSell"]) ], axis=1)
    result['forecast'] = np.where(result['forecastSell'] > 0.5, result['forecastSell'], result['forecastBuy'])
    result['action'] = np.where(result['forecastSell'] > 0.5, 'sell', 'buy')
    result['pnlAbs'] = abs(result['PNL'])



    maxDayOfMonth = result['DayOfMonth'].max()
    #print('maxDayOfMonth='+str(maxDayOfMonth))

     

    monthPnl = 0
    noMonth = 0
    
    '''
    for dm in range(1, maxDayOfMonth+1):
        #print('processing day :'+str(dm))
        #fileOut.write('processing day :'+str(dm) + " \n")

        day_df = result.loc[(result['DayOfMonth'] == dm)]
        if(len(day_df)==0):
            continue

        day_df = day_df.sort_values(by=['forecast'], ascending=False)

        dayPNL = 0
        desc = "Forecast: "

        trade_df = day_df.head(tradeperday)
         
        for index, row in trade_df.iterrows():
            
            if(row["forecastBuy"] > 0.5):
                dayPNL = dayPNL + row["PNL"]
                noMonth = noMonth +1
            elif(row["forecastSell"] > 0.5):
                dayPNL = dayPNL - row["PNL"]
                noMonth = noMonth +1
        monthPnl= monthPnl +dayPNL
        
        #print(str(dm)+". DayBuy="+str(dayPNL)+" monthPnl="+str(monthPnl))
        #print(str(dm)+". Description "+str(desc))
        
    


    ''' 
    noGain = 0
    noLost = 0
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
    if(no==0):
        my_dfG.to_csv('E:\Work\GitHub\Energy\output/my_dfG.txt', sep=';', index=True, mode='a' )


    

    print("-----------------------------------")
    print(str(no)+". noMonth="+str(noMonth)+" noGain="+str(noGain)+" noLost="+str(noLost)+" Accuracy="+str(noGain/noMonth))
    print(str(no)+". monthPnl="+str(monthPnl)+" noMonth="+str(noMonth)+" mean="+str(monthPnl/noMonth))
    totalPNL = totalPNL +monthPnl




# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================
 

#Read data
dfData=read_df(pathGlobal)
print(dfData.head())


print ('len(stockList)=',len(stockList))
time.sleep(3)


dfFit1 = dfData[dfData['Date'] < "2023-07-01" ]
dfTest1 = dfData[(dfData['Date'] >= "2023-07-01") & (dfData['Date'] < "2023-08-01") ]
dfTest2 = dfData[(dfData['Date'] >= "2023-08-01") & (dfData['Date'] < "2023-09-01") ]
dfTest3 = dfData[(dfData['Date'] >= "2023-09-01") & (dfData['Date'] < "2023-10-01") ]
dfTest4 = dfData[(dfData['Date'] >= "2023-10-01") & (dfData['Date'] < "2023-11-01") ]
dfTest5 = dfData[(dfData['Date'] >= "2023-11-01") & (dfData['Date'] < "2023-12-01") ]
dfTest6 = dfData[(dfData['Date'] >= "2023-12-01") & (dfData['Date'] < "2024-01-01") ]

dfTest7 = dfData[(dfData['Date'] >= "2024-01-01") & (dfData['Date'] < "2024-02-01") ]
dfTest8 = dfData[(dfData['Date'] >= "2024-02-01") & (dfData['Date'] < "2024-03-01") ]
dfTest9 = dfData[(dfData['Date'] >= "2024-03-01") & (dfData['Date'] < "2024-04-01") ]
dfTest10 = dfData[(dfData['Date'] >= "2024-04-01") & (dfData['Date'] < "2024-05-01") ]
dfTest11 = dfData[(dfData['Date'] >= "2024-05-01") & (dfData['Date'] < "2024-06-01") ]

dfTestAll = dfData[(dfData['Date'] >= "2023-07-01") & (dfData['Date'] < "2024-06-01") ]

dfFit1 = dfFit1.reset_index() 
dfTest1 = dfTest1.reset_index() 
dfTest2 = dfTest2.reset_index() 
dfTest3 = dfTest3.reset_index() 
dfTest4 = dfTest4.reset_index() 
dfTest5 = dfTest5.reset_index() 
dfTest6 = dfTest6.reset_index() 
dfTest7 = dfTest7.reset_index() 
dfTest8 = dfTest8.reset_index() 
dfTest9 = dfTest9.reset_index() 
dfTest10 = dfTest10.reset_index() 
dfTest11 = dfTest11.reset_index() 

dfTestAll = dfTestAll.reset_index() 


    





colsFeat = []
#colsFeat.append('Hour')
#colsFeat.append('DayOfWeek')
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

#print(colsFeat)
#scaler = StandardScaler()
#features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.2, random_state=42, shuffle=True )
print("X= ", X_train.shape, X_valid.shape)
print("Y= ", Y_train.shape, Y_valid.shape)


modelsName = ["LGBMClassifier_1", "XGBClassifier_1", "CatBoostClassifier_1",  "ExtraTreesClassifier_1",  "RandomForestClassifier_1",  "GradientBoostingClassifier_1", "XGBClassifier_1", "MPV1", "MPV2", "MPV3", "MPV4", "MPV5", "MPV6", "MPV_Good", "LGBMClassifier_2" , "LGBMClassifier_3", "LGBMClassifier_4", "GradientBoostingClassifier_1", "GradientBoostingClassifier_2", "GradientBoostingClassifier_3", "GradientBoostingClassifier_4", "GradientBoostingClassifier_5", "GradientBoostingClassifier_6", "GradientBoostingClassifier_7", "GradientBoostingClassifier_8", "GradientBoostingClassifier_9"]
#loss_function='RMSE'
models = [  LGBMClassifier(verbose=0), 
            XGBClassifier(verbose=0,learning_rate=0.0003, n_estimators=30000, max_depth=12, subsample =0.1 ),
            CatBoostClassifier(verbose=0),
            ExtraTreesClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            XGBClassifier(verbose=0, n_estimators=1000),
            MLPClassifier(max_iter=50),
            MLPClassifier(activation='identity',max_iter=500),
            MLPClassifier(hidden_layer_sizes=(50,30,50), learning_rate_init=0.00001, max_iter=500),
            MLPClassifier(hidden_layer_sizes=(50,100,200), learning_rate_init=0.00001, max_iter=500),
            MLPClassifier(alpha=0.00001, learning_rate_init=0.00001, hidden_layer_sizes=(50,30,50), max_iter=500),
            MLPClassifier(hidden_layer_sizes=(50,100,50), max_iter=500),
            MLPClassifier(activation='identity',max_iter=500),
            LGBMClassifier(verbose=0, learning_rate=0.05, n_estimators=1000, num_leaves=50), 
            LGBMClassifier(verbose=0, learning_rate=0.01, n_estimators=100, num_leaves=50),
            GradientBoostingClassifier(learning_rate=0.8, n_estimators=300),
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=3),
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=5), 
            GradientBoostingClassifier(learning_rate=0.3, n_estimators=300, max_depth=7), 
            GradientBoostingClassifier(learning_rate=0.5, n_estimators=400, max_depth=4), 
            GradientBoostingClassifier(learning_rate=0.8, n_estimators=600, max_depth=4),
            GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, max_depth=4, loss='exponential' ),
            GradientBoostingClassifier(learning_rate=0.01, n_estimators=400, max_depth=4, loss='exponential' )
        ]
 
for i in range(7, 8):
    print('clungu start training no ' +str(i)+' for model '+str(modelsName[i])+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))

    '''
    history = models[i].fit(X_train, Y_train)

    # Plot training and validation loss
    epochs = range(1, len(models[i].train_losses) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].plot(epochs, models[i].train_losses, label='Train Loss')
    if models[i].validation_data is not None:
        ax[0].plot(epochs, models[i].validation_losses, label='Validation Loss')
    ax[0].set_title('MLP Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot training and validation accuracy
    ax[1].plot(epochs, models[i].train_accuracies, label='Train Accuracy')
    if models[i].validation_data is not None:
        ax[1].plot(epochs, models[i].validation_accuracies, label='Validation Accuracy')
    ax[1].set_title('MLP Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()

    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, models[i].predict(X_valid)))
    print('Validation Score : ', models[i].score(X_valid, Y_valid) )

   


    '''
    # Define the MLPClassifier
    #mlp = MLPClassifier( max_iter=1, hidden_layer_sizes=(150,130,100,150), warm_start=True, learning_rate_init=0.00001, random_state=0)
    mlp = MLPClassifier( max_iter=1, hidden_layer_sizes=(150,130,100,150), warm_start=True, learning_rate_init=0.0001, random_state=0)
    #mlp = MLPClassifier( max_iter=1, hidden_layer_sizes=(150,200,200,150), warm_start=True, learning_rate_init=0.001, random_state=0)

    #mlp = MLPClassifier( max_iter=1, hidden_layer_sizes=(200,150,100,150,200), warm_start=True, learning_rate_init=0.0001, random_state=0) not ok



    max_iter = 50
    train_scores = []
    test_scores = []
    losses = []

    for k in range(max_iter):
        mlp.fit(X_train, Y_train)
        train_scores.append(accuracy_score(Y_train, mlp.predict(X_train)))
        test_scores.append(accuracy_score(Y_valid, mlp.predict(X_valid)))
        losses.append(mlp.loss_)

    


    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, mlp.predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, mlp.predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, mlp.predict(X_valid)))
    print('Validation Score : ', mlp.score(X_valid, Y_valid) )

    Y_pred = mlp.predict(X_valid)
    accuracy = accuracy_score(Y_valid, Y_pred)
    f1 = f1_score(Y_valid, Y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")


    checkForecast(dfTestAll, mlp, 0)

    checkForecast(dfTest1, mlp, 1)
    checkForecast(dfTest2, mlp, 2)
    checkForecast(dfTest3, mlp, 3)
    checkForecast(dfTest4, mlp, 4)
    checkForecast(dfTest5, mlp, 5)
    checkForecast(dfTest6, mlp, 6)
    checkForecast(dfTest7, mlp, 7)
    checkForecast(dfTest8, mlp, 8)
    checkForecast(dfTest9, mlp, 9)
    checkForecast(dfTest10, mlp, 10)
    checkForecast(dfTest11, mlp, 11)




    print('================================ SECOND RUN  ================================ '+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))


    for k in range(max_iter):
        mlp.fit(X_train, Y_train)
        train_scores.append(accuracy_score(Y_train, mlp.predict(X_train)))
        test_scores.append(accuracy_score(Y_valid, mlp.predict(X_valid)))
        losses.append(mlp.loss_)

    


    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, mlp.predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, mlp.predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, mlp.predict(X_valid)))
    print('Validation Score : ', mlp.score(X_valid, Y_valid) )

    Y_pred = mlp.predict(X_valid)
    accuracy = accuracy_score(Y_valid, Y_pred)
    f1 = f1_score(Y_valid, Y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")


    checkForecast(dfTestAll, mlp, 0)

    checkForecast(dfTest1, mlp, 1)
    checkForecast(dfTest2, mlp, 2)
    checkForecast(dfTest3, mlp, 3)
    checkForecast(dfTest4, mlp, 4)
    checkForecast(dfTest5, mlp, 5)
    checkForecast(dfTest6, mlp, 6)
    checkForecast(dfTest7, mlp, 7)
    checkForecast(dfTest8, mlp, 8)
    checkForecast(dfTest9, mlp, 9)
    checkForecast(dfTest10, mlp, 10)
    checkForecast(dfTest11, mlp, 11)











    print('================================ THIRD RUN  ================================ '+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))


    for k in range(max_iter):
        mlp.fit(X_train, Y_train)
        train_scores.append(accuracy_score(Y_train, mlp.predict(X_train)))
        test_scores.append(accuracy_score(Y_valid, mlp.predict(X_valid)))
        losses.append(mlp.loss_)

    


    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, mlp.predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, mlp.predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, mlp.predict(X_valid)))
    print('Validation Score : ', mlp.score(X_valid, Y_valid) )

    Y_pred = mlp.predict(X_valid)
    accuracy = accuracy_score(Y_valid, Y_pred)
    f1 = f1_score(Y_valid, Y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")


    checkForecast(dfTestAll, mlp, 0)

    checkForecast(dfTest1, mlp, 1)
    checkForecast(dfTest2, mlp, 2)
    checkForecast(dfTest3, mlp, 3)
    checkForecast(dfTest4, mlp, 4)
    checkForecast(dfTest5, mlp, 5)
    checkForecast(dfTest6, mlp, 6)
    checkForecast(dfTest7, mlp, 7)
    checkForecast(dfTest8, mlp, 8)
    checkForecast(dfTest9, mlp, 9)
    checkForecast(dfTest10, mlp, 10)
    checkForecast(dfTest11, mlp, 11)


    print('================================ FOURTH RUN  ================================ '+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))


    for k in range(max_iter):
        mlp.fit(X_train, Y_train)
        train_scores.append(accuracy_score(Y_train, mlp.predict(X_train)))
        test_scores.append(accuracy_score(Y_valid, mlp.predict(X_valid)))
        losses.append(mlp.loss_)

    


    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, mlp.predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, mlp.predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, mlp.predict(X_valid)))
    print('Validation Score : ', mlp.score(X_valid, Y_valid) )

    Y_pred = mlp.predict(X_valid)
    accuracy = accuracy_score(Y_valid, Y_pred)
    f1 = f1_score(Y_valid, Y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")


    checkForecast(dfTestAll, mlp, 0)

    checkForecast(dfTest1, mlp, 1)
    checkForecast(dfTest2, mlp, 2)
    checkForecast(dfTest3, mlp, 3)
    checkForecast(dfTest4, mlp, 4)
    checkForecast(dfTest5, mlp, 5)
    checkForecast(dfTest6, mlp, 6)
    checkForecast(dfTest7, mlp, 7)
    checkForecast(dfTest8, mlp, 8)
    checkForecast(dfTest9, mlp, 9)
    checkForecast(dfTest10, mlp, 10)
    checkForecast(dfTest11, mlp, 11)

    
    # Plot the accuracy and loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_scores, label='Train Accuracy')
    plt.plot(test_scores, label='Validation Accuracy')
    plt.title('MLP Neuronal Network Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    

     
    '''
    # Train the model and evaluate on the test set
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]
    models[i].fit(X_train, Y_train, eval_set=eval_set)

    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, models[i].predict(X_valid)))
    print('Validation Score : ', models[i].score(X_valid, Y_valid) )

    # Retrieve evaluation results
    results = models[i].evals_result()
    #print(results)

    # Plot training and validation loss
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].plot(x_axis, results['validation_0']['logloss'], label='Train Loss')
    ax[0].plot(x_axis, results['validation_1']['logloss'], label='Test Loss')
    ax[0].set_title(str(modelsName[i])+' Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Log Loss')
    ax[0].legend()

    # Plot training and validation accuracy
    ax[1].plot(x_axis, 1 - np.array(results['validation_0']['logloss']), label='Train Accuracy')
    ax[1].plot(x_axis, 1 - np.array(results['validation_1']['logloss']), label='Test Accuracy')
    ax[1].set_title(str(modelsName[i])+' Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()
    '''


    '''
    totalPNL=0
    totalTrades=0

    checkForecast(dfTest1, models[i], 1)
    checkForecast(dfTest2, models[i], 2)
    checkForecast(dfTest3, models[i], 3)
    checkForecast(dfTest4, models[i], 4)
    checkForecast(dfTest5, models[i], 5)
    checkForecast(dfTest6, models[i], 6)
    checkForecast(dfTest7, models[i], 7)
    checkForecast(dfTest8, models[i], 8)
    checkForecast(dfTest9, models[i], 9)

    #checkForecast(dfTestAll, models[i], 0)

    if maxPNL < totalPNL:
        maxPNL = totalPNL

    print("==> TotalPNL="+str(totalPNL)+" totalTrades="+str(totalTrades)+" mean="+str(totalPNL/totalTrades) + " while current maxPNL="+str(maxPNL))
    print("----------------------------------------------------------------------")
    '''
    
    
    # save the model to disk
    #filename = str(modelsName[i])+ '_model.sav'
    #pickle.dump(models[i], open('output_model/'+filename, 'wb'))

    
    print("Finish model"+str(modelsName[i]) +' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------------------------------")
