import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats.distributions import chi2
import requests
import json

# np.seterr(divide = 'ignore') 

# 將 行情 API 自訂 function 方便後續使用
def fugle_realtime_api(apiType, symbolId, apiToken):
    # 連結 數位沙盒 realtime API 取得最新數據 
    api_path = f"https://api.fintechspace.com.tw/realtime/v0.3/intraday/{apiType}?symbolId={symbolId}&apiToken={apiToken}&limit=500&oddLot=true&jwt={jwt}"
    price_col = requests.get(api_path)
    price_data = price_col.json()['data'][apiType]
    return price_data

# 輸入股票代碼
symbolId = "2330"

# 請輸入自己的數位沙盒 apiToken
apiToken = "4af7c90c0eac7cd5ee3d289f00045bbb"
jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyMDQ1IiwiaWF0IjoxNjQ4MDIzNTE4LCJpc3MiOiJFMDhPb3VKeElXYnJlOW0yWXh6R2d4ZGhZWWJXU1cyMiJ9.0O7ZQSn6YaLJmiXLxdp1SARybxHKs18sqbdXcH3dQ-o"
price_data = fugle_realtime_api("chart", symbolId, apiToken)
#a:一分鐘的成交均價 O:開盤價 h:最高價 l:最低價 c:收盤價 v:成交量 t:每分鐘單位
# print(json.JSONEncoder().encode({
#     # 'solution': price_data,
#     'residual': 1235555
# }))

# Step 1: 取得原始資料及資料處理

#讀檔
data = pd.read_csv(r'C:\xampp\htdocs\var-garch-app-backend\scipy\optimize\py\2027.csv')
df = pd.DataFrame(data, columns= ['年月日','報酬率％'])
df = df.rename({'年月日': 'date', '報酬率％': 'daily_ret'}, axis='columns')
df['daily_ret'] = df['daily_ret'] /100
df[['date']] = df[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[0:4], s[4:6], s[6:]))
df['date'] = pd.to_datetime(df['date'])
df
# print(json.dumps({
#     'solution': json.loads(df.to_json(orient = 'columns')),
#     'residual': 12344
# }))


# Step 2: 模型參數估計

# GARCH(1,1)

########### MLE ####################

#用MLE算參數
def log_likelihood_MLE(params):
    df['variance'] = df['daily_ret']
    df['variance'][0] = 0
    df['variance'][1] = (df['daily_ret'][0])**2
    
    alpha, beta, gamma = params[0], params[1], params[2]
    #alpha, beta, omega = params[0], params[1], params[2]
    
    window = 250   # 估計參數所使用之樣本數
    var_long =  0.0003   #假設的
    
    #df['stdev'] = df['daily_ret'].rolling(window=2, center=False).std(ddof=0)
    
    for i in range(len(df['daily_ret'])):
        if i > 1:
            df['variance'][i] =  gamma * var_long + beta * (df['variance'][i-1]) + alpha * (df['daily_ret'][i-1])**2
            #df['variance'][i] =  omega + beta * (df['stdev'][i-1] )**2 + alpha * (df['daily_ret'][i-1])**2
            df['stdev'] = (df['variance']) ** 0.5
           
    df['loglikelihood'] = -np.log(df['variance']) - ((df['daily_ret'])**2) / df['variance']   #loglikelihood

    df_loglike = df[1:window]

    sums = df_loglike['loglikelihood'].sum()
    return(-sums)

# log_likelihood_MLE([0, 0.92, 0.08])

#規劃求解

bounds = Bounds([0.0, 0.0, 0.0], [1.00, 1.00, 1.00])         #  0 < alpha, beta, gamma < 1

cons = {'type': 'eq', 'fun': lambda params: params[0] + params[1] + params[2] - 1}       # alpha + beta + gamma = 1

#cons = {'type': 'ineq', 'fun': lambda params: -params[0] - params[1] - params[2] + 1}   # alpta + beta  <  1 (另一種)

mle_model = minimize(log_likelihood_MLE, np.array([3, 3, 3]), method='SLSQP', bounds = bounds, constraints = cons)
mle_model
# print(json.dumps({
#     'solution': mle_model,
#     'residual': 12344
# }))


#mle_model.x[0]      #alpha
#mle_model.x[1]      #beta
#mle_model.x[2]      #gamma

# df[:250]


# # # Step 3: 計算變異數及風險值

# window = 250      #實際回測使樣本數 ( = 總樣本數 - 估計參數使用之樣本數)
# var_long =  0.0003

# df['stdev'] = df['daily_ret']
# df['stdev'][window - 1] = 0
# df['stdev'][window] = abs(df['daily_ret'][window - 1])

# for i in range(window, len(df['daily_ret'])):
    
#     #####   計算變異數
#     df['variance'][i] = mle_model.x[2] * var_long + mle_model.x[1] * (df['variance'][i-1]) + mle_model.x[0]  * (df['daily_ret'][i-1])**2
#     df['stdev'] = df['variance'] **0.5   
    

#     #####   計算風險值
#     df['99% VaR']= -2.33 *df['stdev']  #Normal Distribution (99% VaR)
#     #df['99% VaR']= -3.365 * df['stdev']    #Student-t Distribution df=5 (99% VaR)
    
#     #df['95% VaR']= -1.645 *df['stdev']  #Normal Distribution (95% VaR)
#     #df['95% VaR']= -2.015 * df['stdev']    #Student-t Distribution df=5 (95% VaR)
    
#     df['difference']=df['daily_ret'] - df['99% VaR']
#     #df['difference']=df['daily_ret'] - df['95% VaR']
           
# df['stdev'] = df['variance'] **0.5    
# df['stdev_year'] = df['stdev'] * (252**0.5)  #  年化標準差
# #dfs[:200]
# df_new = df[250:]
# df_new = df_new.reset_index()
# df_new

# #exceedance (穿透次數)
# count = 0
# g = 0
# for i in range(len(df_new['daily_ret'])):
#     g+= 1
#     if df_new['difference'][i] < 0:
#         count+=1
# # print("樣本數 =", g, "  " ,"穿透次數 =", count)

# # results = ["樣本數 =", g, "  " ,"穿透次數 =", count]

# # # 繪圖

# # plt.figure(figsize=(14,6))

# # #plt.scatter(df_new['date'], df_new['daily_ret'], c='green')

# # for i in range(len(df_new)):
# #     if df_new['difference'][i] < 0:
# #         plt.scatter(df_new['date'][i], df_new['daily_ret'][i], c = 'blue')
# #     else:
# #         plt.scatter(df_new['date'][i], df_new['daily_ret'][i], c='green')

# # plt.plot(df_new['date'], df_new['99% VaR'], c= "red")

# # plt.show()


# # def garch_filter(alpha0, alpha1, beta, eps):
# #     iT = len(eps)
# #     sigma_2 = np.zeros(iT)
# #     
# #     for i in range(iT):
# #         if i == 0:
# #             sigma_2[i] = alpha0 / (1 - alpha1 - beta)
# #         else:
# #             sigma_2[i] = alpha0 / alpha1 * eps[i-1] ** 2 + beta * sigma_2[i-1]
# #             
# #     return sigma_2

# # def garch_loglike(vP, eps):
# #     iT = len(eps)
# #     alpha0 = vP[0]
# #     alpha1 = vP[1]
# #     beta = vP[2]
# #     
# #     sigma_2 = garch_filter(alpha0, alpha1, beta, eps)
# #     
# #     LogL = -np.sum(-np.log(sigma_2) - eps ** 2 / sigma_2)
# #     
# #     return LogL

# # windows = 20
# # 
# # df['stdev21'] = df['daily_ret'].rolling(window=windows, center=False).std()
# # 
# # for i in range(len(df['daily_ret'])):
# #     if i > windows - 1:
# #         df['stdev21'][i] = (lambdas * (df['stdev21'][i-1])**2 + (1 - lambdas) * (df['daily_ret'][i-1])**2)**0.5
# #         df['99% VaR']=norm.ppf(0.01)*df['stdev21']
# #         df['difference']=df['daily_ret'] - df['99% VaR']
# # 
# # df['hvol21'] = df['stdev21'] * (252**0.5)                    #  年化標準差
# # df['variance'] = df['hvol21']**2                                     # 年化波動率
# #     

# #  # Kupiec Proportion of Failures (POF) Test (1995)

# #Unconditional Coverage Test
# def Kupiec_POF(df, p):
#     exception = 0
#     length = len(df)         
#     for i in range(length):
#         if df['difference'][i] < 0:   #首筆資料沒有difference，所以從第二筆開始
#             exception += 1
#     LR_UC = -2* math.log( ((1-p)**(length-exception)) * (p**exception) ) + 2 * math.log(  ((1 - exception/length)**(length-exception)) *( (exception/length)**exception ))
#     print('Length = ', length)
#     print('Exception = ', exception)
#     print("LR_UC statistics =",  LR_UC)
#     print("p-value = ", chi2.sf( LR_UC, 1) )
#     if chi2.sf( LR_UC, 1) < 0.05:   #95%信心水準
#         print("Reject null hypothesis.")
#     else:
#         print("Do not reject null hypothesis.")

# # Kupiec_POF(df_new, 0.01)


# # # Christoffersen Test (1998)

# #Independence Test
# def Christoffersen_test(df, p):
#     exception = 0
#     length = len(df)         
#     for i in range(length):
#         if df['difference'][i] < 0:   #首筆資料沒有difference，所以從第二筆開始
#             exception += 1
#     LR_UC = -2* math.log( ((1-p)**(length-exception)) * (p**exception) ) + 2 * math.log(  ((1 - exception/length)**(length-exception)) * (exception/length)**exception )

#     a00 = 0
#     a01 = 0
#     a10 = 0
#     a11 = 0
#     a02 = 0
#     for i in range(length-1):
#         if df['difference'][i] > 0 and df['difference'][i+1] > 0:
#             a00 += 1
#         elif df['difference'][i] > 0 and df['difference'][i+1] < 0:
#             a01 += 1
#         elif df['difference'][i] < 0 and df['difference'][i+1] > 0:
#             a10 += 1
#         elif df['difference'][i] < 0 and df['difference'][i+1] < 0:
#             a11 += 1
#         else:
#             a02 += 1
#     q0 = a00 / (a00 + a01)
#     q1 = a10 / (a10 + a11)
#     q = (a00 + a10) / (a00 + a01 + a10 + a11)
#     LR_IND = -2* math.log( ( (1-q)**(a01+a11) )* (q**(a00+a10)) / (( (1-q0)**a01 ) * (q0** a00) * ( (1-q1)**a11 ) * (q1**a10) ))

#     LR_CC = LR_UC + LR_IND

#     #Summary
#     print("LR_UC statistics =",  LR_UC)
#     print("p-value = ", chi2.sf( LR_UC, 1) )
#     print("LR_IND statistics =",  LR_IND)
#     print("p-value = ", chi2.sf( LR_IND, 1) )
#     print("LR_CC statistics =",  LR_CC)
#     print("p-value = ", chi2.sf( LR_CC, 2) )
#     if chi2.sf( LR_CC, 2) < 0.05:
#         print("Reject null hypothesis.")
#     else:
#         print("Do not reject null hypothesis.")

# # Christoffersen_test(df_new, 0.01)



