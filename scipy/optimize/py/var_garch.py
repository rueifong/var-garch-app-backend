#!/usr/bin/env python
# coding: utf-8

# # 風險值(VaR)計算_範例檔

# In[ ]:


import csv
import math
from urllib import response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import t
from scipy.stats.distributions import chi2
from scipy.optimize import minimize
from scipy.optimize import Bounds
import seaborn as sns
from statsmodels.base.model import GenericLikelihoodModel
import json
import sys
import requests


# # Step 1: 取得原始資料及資料處理

# In[ ]:


# 讀檔
# code = input("輸入股票代碼 : ")
code = sys.argv[1]
model = sys.argv[2]
CSV_PATH = sys.argv[3]
LIST_PATH = sys.argv[4]
filename = str(code) + ".csv" 

# 將 行情 API 自訂 function 方便後續使用
# def fugle_realtime_api(apiType, symbolId, apiToken):
#     # 連結 數位沙盒 realtime API 取得最新數據 
#     api_path = f"https://api.fintechspace.com.tw/realtime/v0.3/intraday/{apiType}?symbolId={symbolId}&apiToken={apiToken}&limit=500&oddLot=true&jwt={jwt}"
#     price_col = requests.get(api_path)
#     price_data = price_col.json()['data'][apiType]
#     return price_data

# # 輸入股票代碼
# # symbolId = "2330"

# # 請輸入自己的數位沙盒 apiToken
# apiToken = "4af7c90c0eac7cd5ee3d289f00045bbb"
# jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyMDQ1IiwiaWF0IjoxNjQ4MDIzNTE4LCJpc3MiOiJFMDhPb3VKeElXYnJlOW0yWXh6R2d4ZGhZWWJXU1cyMiJ9.0O7ZQSn6YaLJmiXLxdp1SARybxHKs18sqbdXcH3dQ-o"
# price_data = fugle_realtime_api("chart", code, apiToken)
#a:一分鐘的成交均價 O:開盤價 h:最高價 l:最低價 c:收盤價 v:成交量 t:每分鐘單位


#輸入股票代碼
# symbolId = 2618

#輸入自己的 apiToken
# apiToken ="cf456d821fcdf0aef9c4df112f01be3a"
# # 抓取分k資料api---抓fugle api
# price_col = requests.get(f"https://api.fugle.tw/realtime/v0.3/intraday/chart?symbolId={str(code)}&apiToken={apiToken}")

# # 查看資料格式
# # price_col.json()
# # 轉換為 dataframe 格式
# price_data = pd.DataFrame(price_col.json()['data']['chart'])

# # 轉換 欄位 t 的 timestamp 格式為 datetime 格式
# price_data['t'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), price_data['t']))
#a:一分鐘的成交均價 O:開盤價 h:最高價 l:最低價 c:收盤價 v:成交量 t:每分鐘單位



# df = pd.DataFrame(price_data, columns= ['t','c', 'v'])
# df = df.rename({'t': 'date', 'c': 'price', 'v': 'quantity'}, axis='columns')
# print(json.dumps({
#     'price_data': price_data,
#     'df': json.loads(df.to_json(orient = 'columns')),
#     'residual': 1235555,
#     'residual2': 234,
# }))


# data = pd.read_csv(str(filename))
# data = pd.read_csv(r'2330.csv')
data = pd.read_csv(CSV_PATH + str(filename))
df = pd.DataFrame(data, columns= ['t','c', 'v'])
df = df.rename({'t': 'date', 'c': 'price', 'v': 'quantity'}, axis='columns')
df['daily_ret'] = np.log(df.price) - np.log(df.price.shift(1))
df['date'] = pd.to_datetime(df['date'])
df=df.dropna()

#敘述統計
def descriptive_statistics(x) : 
    return pd.Series([x.mean(),x.std(),x.var(),
                      x.kurt(),x.skew()],index=['mean','std','var','kurtosis','skewness'])
# mean平均數 std標準差 var變異數  kurtosis峰態    skewness偏態

descriptive_statistics_result = pd.DataFrame(df, columns=['daily_ret'])


# # 波動度模型：GARCH(1,1)、EWMA

# # Step 3: 套用上面估計的參數，計算變異數及風險值

# In[ ]:


# 查表取得GARCH模型參數
data_new = pd.read_csv(LIST_PATH)
new = data_new[data_new["code"] == int(code)]

while True:
    # volatility_model = input("請選擇波動度模型(G:GARCH/E:EWMA): ")
    volatility_model = model

    if volatility_model == "G":
        gamma = new["gamma"]
        beta = new["beta"]
        alpha = new["alpha"]
        # print("您選擇的波動度模型為GARCH(1,1) ")
        break

    elif volatility_model == "E":
        gamma = 0.0000000
        beta = 0.9400000
        alpha = 1.0000000 - beta
        # print("您選擇的波動度模型為EWMA ")
        break

    else:
        # print("輸入錯誤，請重新輸入")
        continue

var_long = np.std(df['daily_ret'][1:])** 2     # 長期平均變異數 (此處以所有樣本的標準差取平方)    


# ![download.png](attachment:download.png)

# In[ ]:

######################################################################

# 這邊之後要改餵即時資料，目前先以歷史資料進行回測作為範例
var_alpha = 0.01          #風險值顯著水準(一般常用：0.01, 0.05, 0.10)
window = 40

df['variance'] = df['daily_ret']**2
df['variance'][1] = 0
df['variance'][2] = df['daily_ret'][:2].var()


for i in range(3, window+3):
    
    #####   計算變異數
    df['variance'][i] = gamma * var_long + beta * (df['variance'][i-1]) + alpha  * (df['daily_ret'][i-1])**2
    df['stdev'] = df['variance'] **0.5   
    
#####   計算風險值
df['99% VaR'] = -2.3263478740408408 * df['stdev']    # Normal (99% VaR)
#df['VaR']= -3.3649299989072756 * df['stdev']    # Student-t df=5 (99% VaR)
df['95% VaR'] = -1.6448536269514722 * df['stdev']    # Normal(95% VaR)
#df['VaR']= -2.0150483726691575 * df['stdev']    # Student-t df=5 (95% VaR)
df['90% VaR'] = -1.2815515655446004 * df['stdev']    # Normal(95% VaR)
#df['VaR']= -1.475884048782027 * df['stdev']     # Student-t df=5 (95% VaR)

if var_alpha == 0.01:
    df['VaR'] = df['99% VaR']
elif var_alpha == 0.05:
    df['VaR'] = df['95% VaR']
elif var_alpha == 0.10:
    df['VaR'] = df['90% VaR']    
else:
    df['VaR'] = norm.ppf(var_alpha) * df['stdev']
    #df['VaR'] = t.ppf(var_alpha, 5)

df['difference'] = df['daily_ret'] - df['VaR']
#df['difference']=df['daily_ret'] - df['95% VaR']
           
df['stdev'] = df['variance'] **0.5

df_new = df[:window]
df_new = df_new.reset_index()


####### 模型檢定1：穿透比例檢定
exception = 0
length = 0

for i in range(len(df_new['daily_ret'])):
    length += 1
    if df_new['difference'][i] < 0:
        exception += 1        

# print('回測筆數 = ', length)
# print('穿透次數 = ', exception)

ExceedRatio = round(exception / length, 4)
sigma = (ExceedRatio * (1 - ExceedRatio) / length) ** 0.5    
UBound = round(1.9599639845400545 * sigma + var_alpha, 4)            # 穿透比例的95%信賴區間上界
LBound = round(-1.9599639845400545 * sigma + var_alpha, 4)           # 穿透比例的95%信賴區間下界

if LBound < 0:           #機率沒有負值，故下界最小即為0
    LBound = 0

#print("VaR顯著水準 = ", var_alpha)
#print("95%信賴區間：", LBound,",", UBound)
# print("穿透比例 = ", ExceedRatio)
# print("p-value = ", round(norm.sf(ExceedRatio, var_alpha, sigma), 4))

#根據p-value定義該模型是否恰當
if norm.sf(ExceedRatio, var_alpha, sigma) > 0.10:
    is_ok = ("該模型佳")
elif math.isnan(round(norm.sf(ExceedRatio, var_alpha, sigma), 4)) == True:
    is_ok = ("該模型佳")
elif norm.sf(ExceedRatio, var_alpha, sigma) > 0.05:
    is_ok = ("該模型尚可")
else:
    is_ok = ("該模型不佳")


####### 模型檢定2：Kupiec Unconditional Coverage Test
def Kupiec_POF(df, p):
    exception = 0
    length = len(df)         
    for i in range(length):
        if df['difference'][i] < 0:                     # 首筆資料沒有difference，所以從第二筆開始
            exception += 1
    LR_UC = -2* math.log( ((1-p)**(length-exception)) * (p**exception) ) + 2 * math.log(  ((1 - exception/length)**(length-exception)) *( (exception/length)**exception ))
    # print('回測筆數 = ', length)
    # print('穿透次數 = ', exception)
    # print('穿透比例 = ', round(exception / length, 4))
    # #print("LR_UC =",  round(LR_UC, 4))
    # print("p-value = ", round(chi2.sf(LR_UC, 1), 4))
    # if chi2.sf( LR_UC, 1) < 0.05:                       # 檢定顯著水準 0.05
    #     print("該模型不佳")
    # else:
    #     print("該模型佳")
    return {
        'length': length,
        'exception': exception,
        'exceedRatio': round(exception / length, 4),
        # 'LR_UC': round(LR_UC, 4),
        'p-value': round(chi2.sf(LR_UC, 1), 4),
        'is_ok': "該模型不佳" if chi2.sf( LR_UC, 1) < 0.05 else "該模型佳",
    }


####### 模型檢定3：Christoffersen Test
def Christoffersen_test(df, p):
    exception = 0
    length = len(df)         
    for i in range(length):
        if df['difference'][i] < 0:   #首筆資料沒有difference，所以從第二筆開始
            exception += 1
    LR_UC = -2* math.log( ((1-p)**(length-exception)) * (p**exception) ) + 2 * math.log(  ((1 - exception/length)**(length-exception)) * (exception/length)**exception )

    a00 = 0
    a01 = 0
    a10 = 0
    a11 = 0
    a02 = 0
    for i in range(length-1):
        if df['difference'][i] > 0 and df['difference'][i+1] > 0:
            a00 += 1
        elif df['difference'][i] > 0 and df['difference'][i+1] < 0:
            a01 += 1
        elif df['difference'][i] < 0 and df['difference'][i+1] > 0:
            a10 += 1
        elif df['difference'][i] < 0 and df['difference'][i+1] < 0:
            a11 += 1
        else:
            a02 += 1

    q0 = a00 / (a00 + a01)
    
    if a10 + a11 == 0:
        q1 = 0
    else:
        q1 = a10 / (a10 + a11)
        
    q = (a00 + a10) / (a00 + a01 + a10 + a11)
    LR_IND = -2* math.log( ( (1-q)**(a01+a11) )* (q**(a00+a10)) / (( (1-q0)**a01 ) * (q0** a00) * ( (1-q1)**a11 ) * (q1**a10) ))

    LR_CC = LR_UC + LR_IND

    #Summary
    # print("回測筆數 = ", length)
    # print("穿透次數 = ", exception)
    # print('穿透比例 = ', round(exception / length, 4))
    #print("LR_UC = ",  round(LR_UC, 4))
    #print("p-value = ", chi2.sf( LR_UC, 1) )
    #print("LR_IND = ",  round(LR_IND, 4))
    #print("p-value = ", chi2.sf( LR_IND, 1) )
    #print("LR_CC = ",  round(LR_CC, 4))
    # print("p-value = ", round(chi2.sf(LR_CC, 2), 4))
    if chi2.sf(LR_CC, 2) < 0.05:
        is_ok = ("該模型不佳")
    else:
        is_ok = ("該模型佳")
    return {
        'length': length,
        'exception': exception,
        'exceedRatio': round(exception / length, 4),
        # 'LR_UC': round(LR_UC, 4),
        # 'LR_UC_p-value': chi2.sf( LR_UC, 1),
        # 'LR_IND': round(LR_IND, 4),
        # 'LR_IND_p-value': chi2.sf( LR_IND, 1),
        # 'LR_CC': round(LR_CC, 4),
        'p-value': round(chi2.sf(LR_CC, 2), 4),
        'is_ok': is_ok,
    }

#四分位數
arry = df_new['difference']
percentile_25 = np.percentile(arry, 25,interpolation='linear') 
percentile_50 = np.percentile(arry, 50,interpolation='linear') 
percentile_75 = np.percentile(arry, 75,interpolation='linear')

# print('percentile_25 = ',percentile_25)
# print('percentile_50 = ',percentile_50)
# print('percentile_75 = ',percentile_75)

###### 圖(3)紅綠燈
i = 10
if df_new['difference'][i] < percentile_25:
    light = ("紅燈")
elif df_new['difference'][i] < percentile_50:
    light = ("黃燈")
else:
    light = ("綠燈")

#######圖(3)常態分配圖
mu = 0                                   #平均數
sigma = df['stdev'][10]                  #當期算出來的波動度

xlim = [mu - 3 * sigma, mu + 3 * sigma] 
x = np.linspace(xlim[0], xlim[1], 1000)
y = norm.pdf(x, mu, sigma)

print(json.dumps({
    'code': code,                    # 股票代碼
    'long-run_variance': var_long,   # long-run variance
    'data_length': length,      # 樣本數
    'model_1': {                     # 模型檢定1：比例檢定
        'length': length,  # 樣本數
        'exception': exception,      # 穿透次數
        # 'var_alpha': var_alpha,      # VaR顯著水準
        # 'LBound': LBound,            # 95%信賴區間 LBound
        # 'UBound': UBound,            # 95%信賴區間 UBound
        'exceedRatio': ExceedRatio,  # 實際穿透比例
        'p-value': round(norm.sf(ExceedRatio, var_alpha, sigma), 4),
        'is_ok': is_ok,              # 該模型合適
    },
    'model_2': Kupiec_POF(df_new, var_alpha),            # 模型檢定2：Kupiec POF Test (1995)
    'model_3': Christoffersen_test(df_new, var_alpha),   # 模型檢定3：Christoffersen Test (1998)
    'percentile_25': percentile_25,
    'percentile_50': percentile_50,
    'percentile_75': percentile_75,
    'light': light,
    'normal_graph': {
        'x': x.tolist(),
        'y': y.tolist(),
        'line99': -2.3263478740408408 * sigma,
        'line95': -1.6448536269514722 * sigma,
        'line90': -1.2815515655446004 * sigma,
        # 'line99': norm.ppf(0.01) * sigma,
        # 'line95': norm.ppf(0.05) * sigma,
        # 'line90': norm.ppf(0.10) * sigma,
    },
    'descriptive_statistics': json.loads(descriptive_statistics_result.apply(descriptive_statistics).to_json(orient = 'columns')),
    'df': json.loads(df.to_json(orient = 'columns')),
    'df_new': json.loads(df_new.to_json(orient = 'columns')),
}))

# # 繪圖

# # In[ ]:


# plt.figure(figsize=(14,6))

# #plt.scatter(df_new['date'], df_new['daily_ret'], c='green')

# for i in range(len(df_new)):
#     if df_new['difference'][i] < 0:
#         plt.scatter(df_new['date'][i], df_new['daily_ret'][i], c = 'blue')
#     else:
#         plt.scatter(df_new['date'][i], df_new['daily_ret'][i], c = 'green')

# plt.plot(df_new['date'], df_new['VaR'], c = "red")
# plt.plot(df_new['date'], df_new['99% VaR'], c = "red")
# plt.plot(df_new['date'], df_new['95% VaR'], c = "orange")
# plt.plot(df_new['date'], df_new['90% VaR'], c = "pink")

# plt.show()


# # In[ ]:




#######圖(3)常態分配圖
# mu = 0                                   #平均數
# sigma = df['stdev'][10]                #當期算出來的波動度

# xlim = [mu - 3 * sigma, mu + 3 * sigma] 
# x = np.linspace(xlim[0], xlim[1], 1000)
# y = norm.pdf(x, mu, sigma)
# plt.plot(x, y)
# ax = plt.gca()
# ax.set_facecolor('k')

# plt.grid(True, linestyle='--', which='major')
# plt.axvline(norm.ppf(0.01) * sigma, linestyle='dashed', color='green')        #99% VaR
# plt.axvline(norm.ppf(0.05) * sigma, linestyle='dashed', color='orange')       #95% VaR
# plt.axvline(norm.ppf(0.10) * sigma, linestyle='dashed', color='red')          #90% VaR

# plt.title('Distribution Graph')        #圖表標題
# #plt.title('Normal({}, {})'.format(mu, round(sigma, 4)))      
# plt.show()