import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import catboost as cb
import time
import pickle 
import warnings

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, neighbors

from sklearn.metrics import mean_squared_error
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


pathGlobal = r'D:\work\PretunnelSearch\stock\stock_data\data'  
pathTest1 = r'D:\work\PretunnelSearch\stock\stock_data\data_202401' 
pathTest2 = r'D:\work\PretunnelSearch\stock\stock_data\data_202402' 

pathTest1b = r'D:\work\PretunnelSearch\stock\stock_data\data_202309'
pathTest2b = r'D:\work\PretunnelSearch\stock\stock_data\data_202310'
pathTest3b = r'D:\work\PretunnelSearch\stock\stock_data\data_202311'
pathTest4b = r'D:\work\PretunnelSearch\stock\stock_data\data_202312'


stockList=  ['SPY', 'TSLA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'EEM', 'INTC', 'KVUE', 'SLV', 'DIS', 'BABA', 'TLT', 'PLTR', 'GOOGL', 'BAC', 'NFLX', 'C', 'XLF', 'WBD', 'ROKU', 'GOOG', 'PYPL', 'PFE', 'F', 'SQ', 'JNJ', 'XOM', 'SOXL', 'COIN', 'AFRM', 'OXY', 'EWZ', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SNAP', 'SHOP', 'QCOM', 'JPM', 'PARA', 'WBA', 'T', 'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'AVGO', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'ENPH', 'CVX', 'VALE', 'DG', 'DIA', 'CHWY', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'PINS', 'CRWD', 'VZ', 'KO', 'WMT', 'ET', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'LYFT', 'GTLB', 'CVS', 'MRNA', 'U', 'FSLR', 'ETSY', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'HOOD', 'GS', 'LUV', 'MS', 'PTON', 'SBUX', 'ADBE', 'CCJ', 'DAL', 'RH', 'PEP', 'MDB', 'RIG', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'AEO', 'COST', 'UNH', 'NU']
removeList=['NA']


#Read data from directory into a dataframe and transform
def read_df(pathD):
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']



    #-------------------------------------------- read index SPY
    print("Start reading SPY dataframe ")
    filePath = pathD+"\\SPY.txt"
    spy_df = pd.read_csv(filePath, header=None, names=cols)

    for i in range(0, pretunnelLenght):
        spy_df['Open_'+str(i)+'_orig']=spy_df['Open'].shift(-i)
        spy_df['High_'+str(i)+'_orig']=spy_df['High'].shift(-i)
        spy_df['Low_'+str(i)+'_orig']=spy_df['Low'].shift(-i)
        spy_df['Close_'+str(i)+'_orig']=spy_df['Close'].shift(-i)
        spy_df['Volume_'+str(i)+'_orig']=spy_df['Volume'].shift(-i)

    print("after raw read spy_df.shape=", spy_df.shape)
    print(spy_df.head(10))
    spy_df=spy_df.dropna().reset_index(drop=True)

    for i in range(0, pretunnelLenght):
        spy_df['Open_SPY_'+str(i)]=spy_df['Open_'+str(i)+'_orig']*format//spy_df['Open']
        spy_df['High_SPY_'+str(i)]=spy_df['High_'+str(i)+'_orig']*format//spy_df['Open']
        spy_df['Low_SPY_'+str(i)]=spy_df['Low_'+str(i)+'_orig']*format//spy_df['Open']
        spy_df['Close_SPY_'+str(i)]=spy_df['Close_'+str(i)+'_orig']*format//spy_df['Open']
        spy_df['Volume_SPY_'+str(i)]=spy_df['Volume_'+str(i)+'_orig']*format//spy_df['Volume']

        spy_df = spy_df.drop(['Open_'+str(i)+'_orig'], axis=1)
        spy_df = spy_df.drop(['High_'+str(i)+'_orig'], axis=1)
        spy_df = spy_df.drop(['Low_'+str(i)+'_orig'], axis=1)
        spy_df = spy_df.drop(['Close_'+str(i)+'_orig'], axis=1)
        spy_df = spy_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

    spy_df = spy_df.drop(['Open'], axis=1)
    spy_df = spy_df.drop(['High'], axis=1)
    spy_df = spy_df.drop(['Low'], axis=1)
    spy_df = spy_df.drop(['Close'], axis=1)
    spy_df = spy_df.drop(['Volume'], axis=1)

    print("after transform spy_df.shape=", spy_df.shape)
    #print(spy_df.head(10))




     #-------------------------------------------- read index DIA
    print("Start reading DIA dataframe ")
    filePath = pathD+"\\DIA.txt"
    dia_df = pd.read_csv(filePath, header=None, names=cols)

    for i in range(0, pretunnelLenght):
        dia_df['Open_'+str(i)+'_orig']=dia_df['Open'].shift(-i)
        dia_df['High_'+str(i)+'_orig']=dia_df['High'].shift(-i)
        dia_df['Low_'+str(i)+'_orig']=dia_df['Low'].shift(-i)
        dia_df['Close_'+str(i)+'_orig']=dia_df['Close'].shift(-i)
        dia_df['Volume_'+str(i)+'_orig']=dia_df['Volume'].shift(-i)

    print("after raw read dia_df.shape=", dia_df.shape)
    print(dia_df.head(10))
    dia_df=dia_df.dropna().reset_index(drop=True)

    for i in range(0, pretunnelLenght):
        dia_df['Open_DIA_'+str(i)]=dia_df['Open_'+str(i)+'_orig']*format//dia_df['Open']
        dia_df['High_DIA_'+str(i)]=dia_df['High_'+str(i)+'_orig']*format//dia_df['Open']
        dia_df['Low_DIA_'+str(i)]=dia_df['Low_'+str(i)+'_orig']*format//dia_df['Open']
        dia_df['Close_DIA_'+str(i)]=dia_df['Close_'+str(i)+'_orig']*format//dia_df['Open']
        dia_df['Volume_DIA_'+str(i)]=dia_df['Volume_'+str(i)+'_orig']*format//dia_df['Volume']

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




    #-------------------------------------------- read data
    print("Start reading files to dataframe from ", path)

    dfs = list()
    for x in range(len(stockList)):
        if(stockList[x] not in removeList):
            filePath = pathD+"\\"+stockList[x]+".txt"
            if path.exists(filePath):
                data = pd.read_csv(filePath, header=None, names=cols)
                data['File'] = stockList[x]
                dfs.append(data)
    my_df = pd.concat(dfs, ignore_index=True)
    
    
    my_df['target'] = np.where(my_df['Close'].shift(tunnelLenght) > my_df['Close'], 1, 0)

    print("after raw read my_df.shape=", my_df.shape)
    print(my_df.head(10))

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)+'_orig']=my_df['Open'].shift(-i)
        my_df['High_'+str(i)+'_orig']=my_df['High'].shift(-i)
        my_df['Low_'+str(i)+'_orig']=my_df['Low'].shift(-i)
        my_df['Close_'+str(i)+'_orig']=my_df['Close'].shift(-i)
        my_df['Volume_'+str(i)+'_orig']=my_df['Volume'].shift(-i)
    my_df['Close_sale_orig']=my_df['Close'].shift(tunnelLenght)
    my_df['Date_sale_orig']=my_df['Date'].shift(tunnelLenght)


    print(my_df.head(10))
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


    my_df = pd.merge(my_df, spy_df, on=['Date'], how='inner' )

    print("after merge spy my_df.shape=", my_df.shape)
    #print(my_df.head(10))


    my_df = pd.merge(my_df, dia_df, on=['Date'], how='inner' )

    print("after merge dia my_df.shape=", my_df.shape)
    #print(my_df.head(10))

    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    #my_df['Volume_0']=100
    #for i in range(1, pretunnelLenght):
    #    my_df['Volume_'+str(i)] = np.where(my_df['Volume_'+str(i)+'_orig'] > my_df['Volume_'+str(i-1)+'_orig'], my_df['Volume_'+str(i-1)]+1,my_df['Volume_'+str(i-1)]-1 )





    my_df[['dateOnly', 'time']] = my_df['Date'].str.split('T', n=1, expand=True)
    #print(my_df.head(10))
    my_df[['Hour', 'minute']] = my_df['time'].str.split(':', n=1, expand=True)
    my_df[['year', 'Month', 'day']] = my_df['dateOnly'].str.split('-', n=2, expand=True)
    #print(my_df.head(10))

    ''' 
    my_df['Hour'] = my_df['Hour'].astype('int')
    my_df['Month'] = my_df['Month'].astype('int')

    my_df['Month'] = my_df['Month'] % 3


    my_df.loc[( (my_df["dateOnly"] < "2020-03-07") |
                    ( (my_df["dateOnly"] > "2020-11-01") & (my_df["dateOnly"] < "2021-03-13") ) |
                    ( (my_df["dateOnly"] > "2021-11-06") & (my_df["dateOnly"] < "2022-03-12") ) |
                    ( (my_df["dateOnly"] > "2022-11-05") & (my_df["dateOnly"] < "2023-03-10") ) |
                    ( my_df["dateOnly"] > "2023-11-04") ),   "Hour" ] = my_df['Hour'] -1

    print(my_df.head(10))

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
    print("dfTest.shape after drop my_df[time] > 20:30:00Z", my_df.shape)

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
    print("dfTest.shape after drop my_df[time] > 19:30:00Z", my_df.shape)

    my_df.drop(my_df[my_df["dateOnly"] != my_df["dateOnly_sale_orig"]].index , axis=0, inplace=True)
    print("dfTest.shape after drop dateOnly!=dateOnly_sale_orig", my_df.shape)

    #CLUNGU 
    if pathD==pathGlobal:
        my_df.drop(my_df[my_df["dateOnly"] > "2023-09-01"].index , axis=0, inplace=True)
        print("CLUNGU DROP dfTest.shape after drop dateOnly> 2023-09-01", my_df.shape)


    my_df = my_df.drop(["dateOnly_sale_orig"], axis=1)
    my_df = my_df.drop(["Date_sale_orig"], axis=1)
    my_df = my_df.drop(["Date"], axis=1)
    my_df = my_df.drop(["Open"], axis=1)
    my_df = my_df.drop(["High"], axis=1)
    my_df = my_df.drop(["Low"], axis=1)
    my_df = my_df.drop(["Close"], axis=1)
    my_df = my_df.drop(["Close_sale_orig"], axis=1)
    my_df = my_df.drop(["minute"], axis=1)
    my_df = my_df.drop(["year"], axis=1)
    my_df = my_df.drop(["day"], axis=1)

    '''
    aggHour2 = my_df.groupby(['Hour']).agg(CountHours=('Month', 'count') ).reset_index()
    print(aggHour2.head(20))
    aggMonth2 = my_df.groupby(['Month']).agg(CountMounth=('Hour', 'count') ).reset_index()
    print(aggMonth2.head(20))
    '''


    my_df = my_df.reset_index() 

    print(my_df.head(10))

    print("Finish reading files to dataframe from ", path)
    return my_df



def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    dfTestFeat = dfTest[colsFeat].copy()
    #forecasts = model.predict(dfTestFeat)
    forecasts = model.predict_proba(dfTestFeat)

    #result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecast"]) ], axis=1)
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=[ "not_forecast", "forecast"]) ], axis=1)
    #print(result.head)
    

    buyResult = result.nlargest(2000, "forecast")
    sellResult = result.nsmallest(2000, "forecast")

    #print(buyResult.head)
    #print(buyResult.head)

    totalBuy = 0
    meanBuy = 0
    for k in range(len(buyResult)):
        totalBuy = totalBuy + buyResult["PNL"].iloc[k]
    meanBuy = totalBuy /len(buyResult)

    totalSell = 0
    meanSell = 0
    for k in range(len(sellResult)):
        totalSell = totalSell - sellResult["PNL"].iloc[k]
    meanSell = totalSell /len(sellResult)

    #print("-----------------------------------")
    print(str(no)+". TotalBuy="+str(totalBuy)+" meanBuy="+str(meanBuy))
    print(str(no)+". TotalSell="+str(totalSell)+" meanSell="+str(meanSell))
    totalPNL = totalPNL +totalBuy
    totalPNL = totalPNL +totalSell



# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================
 

#Read data
df=read_df(pathGlobal)
df.head()

dfTest=read_df(pathTest1)
#dfTest= dfTest.reset_index()
dfTest2=read_df(pathTest2)
#dfTest2= dfTest2.reset_index()

dfTest1b=read_df(pathTest1b)
dfTest2b=read_df(pathTest2b)
dfTest3b=read_df(pathTest3b)
dfTest4b=read_df(pathTest4b)






colsFeat = []
#colsFeat.append('Hour')
#colsFeat.append('Month')
for i in range(0, pretunnelLenght):
    colsFeat.append('Open_'+str(i))
    colsFeat.append('High_'+str(i))
    colsFeat.append('Low_'+str(i))
    colsFeat.append('Close_'+str(i))
    #colsFeat.append('Volume_'+str(i))
for i in range(0, pretunnelLenght):
    colsFeat.append('Open_SPY_'+str(i))
    colsFeat.append('High_SPY_'+str(i))
    colsFeat.append('Low_SPY_'+str(i))
    colsFeat.append('Close_SPY_'+str(i))
    #colsFeat.append('Volume_SPY_'+str(i))
for i in range(0, pretunnelLenght):
    colsFeat.append('Open_DIA_'+str(i))
    colsFeat.append('High_DIA_'+str(i))
    colsFeat.append('Low_DIA_'+str(i))
    colsFeat.append('Close_DIA_'+str(i))
    #colsFeat.append('Volume_DIA_'+str(i))

features = df[colsFeat]
target = df['target']


 
#scaler = StandardScaler()
#features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=104, shuffle=True )
print("X= ", X_train.shape, X_valid.shape)
print("Y= ", Y_train.shape, Y_valid.shape)







modelsName = ["CatBoostRegressor", "XGBRegressor",   "XGBRegressor_basic", "LGBMRegressor", "ExtraTreesRegressor", "BaggingRegressor", "RandomForestRegressor", "AdaBoostRegressor", "KNeighborsRegressor", "DecisionTreeRegressor", "XGBRegressor1", "XGBRegressor2"]

models = [ AdaBoostClassifier(  ),
            AdaBoostClassifier( n_estimators=100, learning_rate=0.01, random_state=20160703),
            AdaBoostClassifier( n_estimators=100, learning_rate=0.05, random_state=20160703),
            AdaBoostClassifier( n_estimators=200, learning_rate=0.01),
            AdaBoostClassifier( n_estimators=200, learning_rate=0.1),
            AdaBoostClassifier( n_estimators=300, learning_rate=0.5),
            AdaBoostClassifier(n_estimators=300,learning_rate=1), 
            GradientBoostingClassifier(),
            GradientBoostingClassifier(n_estimators = 200),
            GradientBoostingClassifier(n_estimators = 200, learning_rate= 0.05, criterion='friedman_mse' ),
            GradientBoostingClassifier(n_estimators = 200, learning_rate= 0.05, criterion= 'squared_error' ),
            GradientBoostingClassifier(n_estimators = 300, learning_rate= 0.05),
            GradientBoostingClassifier(n_estimators = 300, learning_rate= 0.001),
            GradientBoostingClassifier(n_estimators = 500, learning_rate= 0.05)
        ]
 
for i in range(15):
    print('clungu start training no ' +str(i)+' for model '+str(modelsName[i])+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))

    models[i].fit(X_train, Y_train)
    

    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict(X_valid)))
    print('Validation Score : ', models[i].score(X_valid, Y_valid) )
    

    totalPNL=0

    checkForecast(dfTest, models[i], 1)
    checkForecast(dfTest2, models[i], 2)

    checkForecast(dfTest1b, models[i], 11)
    checkForecast(dfTest2b, models[i], 12)
    checkForecast(dfTest3b, models[i], 13)
    checkForecast(dfTest4b, models[i], 14)

    if maxPNL < totalPNL:
        maxPNL = totalPNL

    print("==> TotalPNL="+str(totalPNL)+" while current maxPNL="+str(maxPNL))
    print("----------------------------------------------------------------------")

    

    # save the model to disk
    #filename = str(modelsName[i])+ '_model.sav'
    #pickle.dump(models[i], open(filename, 'wb'))

    
    print("Finish model", modelsName[i] +' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------------------------------")