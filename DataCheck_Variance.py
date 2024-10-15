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

# importing statistics module
from statistics import variance


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




stockList = [ "PTON", "W", "RIG", "U", "DKNG", "CHWY", "PATH" , "PLTR", "SNAP", "RH","LYFT", "PENN", "SNOW", "SE", "CLF",
            "XLE", "INTC", "KVUE", "BAC",
            "C", "WBD", "PYPL", "F", "XOM", "EWZ", "GM", "AAL", "PARA", "WBA", "CCL", "PBR",
            "WFC", "VALE", "DG", "NEE", "FCX", "PINS", "ET", "DVN", "UAL",
            "LLY", "LUV", "DAL", "MRVL", "AEO" ]



indexList=[ 'SPY', 'XLK', 'XLP']
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

    print(my_df)

    print("Finish reading files to dataframe from ", path)
    return my_df




def checkForecast(dfTest, model, no):
    global maxPNL
    global totalPNL
    global stockList

    colsFeat = []


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


    return "l"



# ================================================================================
# ================================== START MAIN ==================================
# ================================================================================


dateList = [ "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", "2021-12-01", 
            "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
            "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01", 
            "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"]

#Read data
dfData=read_df(pathGlobal, indexList)


for k in range(18):

            
    #print("current len(stockList="+str(len(stockList)))


    dateStartTraning = dateList[k]
    dateEndTraning = dateList[k+12*2]
    dateStartTest = dateList[k+12*2]
    dateEndTest = dateList[k+12*2+1]

    print(" ->> Check mounth : Traning : "+str(dateStartTraning)+" --> "+str(dateEndTraning)+"  Testing : "+str(dateStartTest)+" --> "+str(dateEndTest))

    dfFit1 = dfData[(dfData['Date'] >= dateStartTraning) & (dfData['Date'] < dateEndTraning) ]
    dfTest1 = dfData[(dfData['Date'] >= dateStartTest) & (dfData['Date'] < dateEndTest) ]

    dfFit1 = dfFit1.sort_values(by=['Date'], ascending=True)
    dfTest1 = dfTest1.sort_values(by=['Date'], ascending=True)

    print("Variance of dfFit1 is % s " %(variance(dfFit1['PNL'])))
    print("Variance of dfTest1 is % s " %(variance(dfTest1['PNL'])))






dfData['pnlAbs'] = abs(dfData['PNL'])

my_dfG = dfData.groupby(["File"]).agg(PNL=('PNL', 'sum') , Trades=('PNL', 'count') , pnlAbsM=('pnlAbs', 'mean')  ).reset_index()
my_dfG[ "PNLPerTrade"] =  my_dfG["PNL"] / my_dfG["Trades"]
my_dfG = my_dfG.sort_values(by=['pnlAbsM'], ascending=True)

print(my_dfG.to_string())

