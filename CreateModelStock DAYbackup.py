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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
 
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
pretunnelLenght=3
tunnelLenght=1

tradeperday=100

maxPNL=0
totalPNL=0


pathGlobal = r'E:\common\stock\data_DAY'  
pathTest1 = r'E:\common\stock\stock_data\data_202401' 
pathTest2 = r'E:\common\stock\stock_data\data_202402' 

pathTest1b = r'E:\common\stock\stock_data\oldAlpaca\data_202309'
pathTest2b = r'E:\common\stock\stock_data\oldAlpaca\data_202310'
pathTest3b = r'E:\common\stock\stock_data\oldAlpaca\data_202311'
pathTest4b = r'E:\common\stock\stock_data\oldAlpaca\data_202312'


#stockList=  [ 'TSLA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'EEM', 'INTC', 'KVUE', 'SLV', 'DIS', 'BABA', 'TLT', 'PLTR', 'GOOGL', 'BAC', 'NFLX', 'C', 'XLF', 'WBD', 'ROKU', 'GOOG', 'PYPL', 'PFE', 'F', 'SQ', 'JNJ', 'XOM', 'SOXL', 'COIN', 'AFRM', 'OXY', 'EWZ', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SNAP', 'SHOP', 'QCOM', 'JPM', 'PARA', 'WBA', 'T', 'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'AVGO', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'ENPH', 'CVX', 'VALE', 'DG', 'DIA', 'CHWY', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'PINS', 'CRWD', 'VZ', 'KO', 'WMT', 'ET', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'LYFT', 'GTLB', 'CVS', 'MRNA', 'U', 'FSLR', 'ETSY', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'HOOD', 'GS', 'LUV', 'MS', 'PTON', 'SBUX', 'ADBE', 'CCJ', 'DAL', 'RH', 'PEP', 'MDB', 'RIG', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'AEO', 'COST', 'UNH', 'NU']
stockList=  ['SPY', 'DIA', 'IWM', 'QQQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'MRNA', 'AAPL', 'NVDA', 'AMD', 'MSFT', 'SLV', 'DIS', 
             'BABA', 'TLT', 'GOOGL', 'NFLX', 'ROKU', 'GOOG', 'SQ', 'JNJ', 'XOM', 'COIN', 'AFRM', 'OXY', 'DOCU', 'GM', 'CVNA', 'AAL', 'ORCL', 'BA', 'DKNG', 'SHOP', 'JPM', 'WBA', 'T', 
             'CCL', 'ZM', 'PBR', 'UBER', 'SNOW', 'TSM', 'ZS', 'NKE', 'WFC', 'EFA', 'TGT', 'SCHW', 'XLU', 'RBLX', 'SE', 'CVX', 'VALE', 'DG', 'NEE', 'FCX', 'BX', 'LULU', 'CRM', 'CRWD', 
             'VZ', 'KO', 'WMT', 'UPS', 'DVN', 'UAL', 'DELL', 'PATH', 'SMCI', 'CVS', 'FSLR', 'LLY', 'KR', 'BMY', 'PENN', 'CLF', 'NEM', 'GS', 'LUV', 'MS', 'SBUX', 'ADBE', 
             'PEP', 'TMUS', 'MRVL', 'X', 'CMCSA', 'TSEM', 'LVS', 'BIDU', 'W', 'OKTA', 'COST', 'UNH']
#removed   DELL BAC PYPL RIOT SNAP PARA AEO PFE EEM CHWY C CCJ F INTC WBD KVUE DAL PLTR ENPH ETSY HOOD TSLA RH  SOXL LYFT MARA MDB PINS U

#indexList=['SPY', 'IWM', 'QQQ']
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
    print(my_df.head(10))

    for i in range(0, pretunnelLenght):
        my_df['Open_'+str(i)+'_orig']=my_df['Open'].shift(-i)
        my_df['High_'+str(i)+'_orig']=my_df['High'].shift(-i)
        my_df['Low_'+str(i)+'_orig']=my_df['Low'].shift(-i)
        my_df['Close_'+str(i)+'_orig']=my_df['Close'].shift(-i)
        #my_df['Volume_'+str(i)+'_orig']=my_df['Volume'].shift(-i)
    my_df['Open_sale_orig']=my_df['Open'].shift(tunnelLenght+1)
    my_df['Close_sale_orig']=my_df['Close'].shift(tunnelLenght)
    my_df['Open_buy_orig']=my_df['Open'].shift(1)
    my_df['Date_sale_orig']=my_df['Date'].shift(tunnelLenght+1)


    print(my_df.head(10))
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
            #dia_df['Volume_'+str(i)+'_orig']=dia_df['Volume'].shift(-i)

        print("after raw read for stock %s has index_df.shape=",indexList[x], dia_df.shape)
        print(dia_df.head(10))
        dia_df=dia_df.dropna().reset_index(drop=True)

        for i in range(0, pretunnelLenght):
            dia_df['Open_'+indexList[x]+'_'+str(i)]=dia_df['Open_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['High_'+indexList[x]+'_'+str(i)]=dia_df['High_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Low_'+indexList[x]+'_'+str(i)]=dia_df['Low_'+str(i)+'_orig']*format//dia_df['Open']
            dia_df['Close_'+indexList[x]+'_'+str(i)]=dia_df['Close_'+str(i)+'_orig']*format//dia_df['Open']
            #dia_df['Volume_'+indexList[x]+'_'+str(i)]=dia_df['Volume_'+str(i)+'_orig']*format//dia_df['Volume']

            dia_df = dia_df.drop(['Open_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['High_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Low_'+str(i)+'_orig'], axis=1)
            dia_df = dia_df.drop(['Close_'+str(i)+'_orig'], axis=1)
            #dia_df = dia_df.drop(['Volume_'+str(i)+'_orig'], axis=1)

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

    print(my_df.columns)
    print(my_df.head(10))





    #my_df['PNL_orig']=my_df['Open_sale_orig']-my_df['Open_buy_orig']
    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Close']
    my_df['PNL_orig']=my_df['Close_sale_orig']-my_df['Open_buy_orig']
    
    my_df['PNL']=my_df['PNL_orig']*format/my_df['Open']
    my_df['PNL']=my_df['PNL'].round(2)

    #my_df['target'] = np.where(my_df['PNL'] > 5, 2, np.where(my_df['PNL'] < -5, 0, 1))
    my_df['target'] = np.where(my_df['PNL'] > 0, 1, 0)

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
    result = pd.concat([dfTest, pd.DataFrame(forecasts, columns=["forecastSell", "forecastBuy"]) ], axis=1)
    result['forecast'] = np.where(result['forecastSell'] > 0.5, result['forecastSell'], result['forecastBuy'])
    result['action'] = np.where(result['forecastSell'] > 0.5, 'sell', 'buy')



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

        

        #print("dm=",dm)
        #print(day_df)
        #time.sleep(3)

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
    for k in range(len(result)):
            
        if(result["forecastSell"].iloc[k] > 0.6):
            result["PNL"].iloc[k] = - result["PNL"].iloc[k]
            monthPnl = monthPnl + result["PNL"].iloc[k]
            noMonth = noMonth +1
        elif(result["forecastBuy"].iloc[k] > 0.6):
            monthPnl = monthPnl + result["PNL"].iloc[k]
            noMonth = noMonth +1
    
     



    my_dfG = result.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count')  ).reset_index()
    my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
    my_dfG = my_dfG.sort_values(by=['PNLPerTrade'], ascending=True)
    #print(my_dfG)

    

    #print("-----------------------------------")
    print(str(no)+". monthPnl="+str(monthPnl)+" noMonth="+str(noMonth))
    totalPNL = totalPNL +monthPnl




# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================
 

#Read data
dfData=read_df(pathGlobal)
print(dfData.head())





dfFit1 = dfData[dfData['Date'] < "2024-01-01" ]

dfTest1 = dfData[(dfData['Date'] >= "2024-01-01") & (dfData['Date'] < "2024-02-01") ]
dfTest2 = dfData[(dfData['Date'] >= "2024-02-01") & (dfData['Date'] < "2024-03-01") ]
dfTest3 = dfData[(dfData['Date'] >= "2024-03-01") & (dfData['Date'] < "2024-04-01") ]
dfTest4 = dfData[(dfData['Date'] >= "2024-04-01") & (dfData['Date'] < "2024-05-01") ]
dfTest5 = dfData[(dfData['Date'] >= "2024-05-01") & (dfData['Date'] < "2024-06-01") ]
dfTest6 = dfData[(dfData['Date'] >= "2024-06-01") & (dfData['Date'] < "2024-06-13") ]

dfTestAll = dfData[(dfData['Date'] >= "2023-07-01") & (dfData['Date'] < "2024-06-13") ]

dfFit1 = dfFit1.reset_index() 
dfTest1 = dfTest1.reset_index() 
dfTest2 = dfTest2.reset_index() 
dfTest3 = dfTest3.reset_index() 
dfTest4 = dfTest4.reset_index() 
dfTest5 = dfTest5.reset_index() 
dfTest6 = dfTest6.reset_index() 

dfTestAll = dfTestAll.reset_index() 


    





colsFeat = []
#colsFeat.append('Hour')
#colsFeat.append('Month')
for i in range(0, pretunnelLenght):
    colsFeat.append('Open_'+str(i))
    colsFeat.append('High_'+str(i))
    colsFeat.append('Low_'+str(i))
    colsFeat.append('Close_'+str(i))
    #colsFeat.append('Volume_'+str(i))
for x in indexList:
    for i in range(0, pretunnelLenght):
        colsFeat.append('Open_'+x+'_'+str(i))
        colsFeat.append('High_'+x+'_'+str(i))
        colsFeat.append('Low_'+x+'_'+str(i))
        colsFeat.append('Close_'+x+'_'+str(i))
        #colsFeat.append('Volume_SPY_'+str(i))


features = dfFit1[colsFeat]
target = dfFit1['target']


#scaler = StandardScaler()
#features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=104, shuffle=True )
print("X= ", X_train.shape, X_valid.shape)
print("Y= ", Y_train.shape, Y_valid.shape)


modelsName = ["LGBMClassifier_1", "XGBClassifier_1", "CatBoostClassifier_1",  "ExtraTreesClassifier_1",  "RandomForestClassifier_1",  "GradientBoostingClassifier_1", "XGBClassifier_2", "XGBClassifier_3", "XGBClassifier_4", "LGBMClassifier_2" , "LGBMClassifier_3", "LGBMClassifier_4", "GradientBoostingClassifier_1", "GradientBoostingClassifier_2", "GradientBoostingClassifier_3", "GradientBoostingClassifier_4", "GradientBoostingClassifier_5", "GradientBoostingClassifier_6", "GradientBoostingClassifier_7", "GradientBoostingClassifier_8", "GradientBoostingClassifier_9"]
#loss_function='RMSE'
models = [  LGBMClassifier(verbose=0), 
            XGBClassifier(verbose=0),
            CatBoostClassifier(verbose=0),
            ExtraTreesClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            XGBClassifier(verbose=0, iterations=10000, learning_rate=0.08),
            XGBClassifier(max_depth=5, learning_rate=0.2),
            XGBClassifier(max_depth=6, iterations=10000, learning_rate=0.3),
            LGBMClassifier(verbose=0, n_estimators=2000, learning_rate=0.8),
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
 
for i in range(0, 5):
    print('clungu start training no ' +str(i)+' for model '+str(modelsName[i])+' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))

    models[i].fit(X_train, Y_train)
    

    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict(X_valid)))
    print('Log Lost : ', metrics.log_loss(Y_valid, models[i].predict(X_valid)))
    
    print('Validation Score : ', models[i].score(X_valid, Y_valid) )
    

    totalPNL=0

    checkForecast(dfTest1, models[i], 1)
    checkForecast(dfTest2, models[i], 2)
    checkForecast(dfTest3, models[i], 3)
    checkForecast(dfTest4, models[i], 4)
    checkForecast(dfTest5, models[i], 5)
    checkForecast(dfTest6, models[i], 6)

    #checkForecast(dfTestAll, models[i], 0)

    if maxPNL < totalPNL:
        maxPNL = totalPNL

    print("==> TotalPNL="+str(totalPNL)+" while current maxPNL="+str(maxPNL))
    print("----------------------------------------------------------------------")

    
    
    # save the model to disk
    filename = str(modelsName[i])+ '_model.sav'
    pickle.dump(models[i], open('output_model/'+filename, 'wb'))

    
    print("Finish model", modelsName[i] +' at time  '+datetime.datetime.now().strftime("%H:%M:%S"))
    print("-------------------------------------------------------")
    time.sleep(30)