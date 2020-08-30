# encoding: UTF-8

from datetime import datetime, timedelta
from common import *
from price import DailyPrice, Valuation, Security
from USprice import *
from finance import IncomeQuarter, BalanceQuarter
from indicators import IsST, MoneyFlow, indicators
from industry import Industry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import algo.timeseries

import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode
import plotly.io as pio
import plotly
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
import statsmodels.api as sm

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)


class arch(object):
    def __init__(self):  
        self.start_date = "2016-01-01"
        self.end_date = TODAY
    
    def arch(self, index="GLD"):
        self.start_date = "2019-01-03"
        self.end_date = TODAY
        if isChinaMkt(index):
            self.df = DailyPrice().load(security=index)
        else:
            self.df = USDailyPrice().load(security=index)
            
        df = self.df.loc[
            (self.df.index > self.start_date)
            & (self.df.index <= self.end_date)
        ]
        
        #adf test for log return series to see if stationary
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df['rtn'] = df['close'].pct_change()
        df = df[['close', 'log_rtn','rtn']].dropna(how = 'any')   
        X = df["rtn"]
        result = adfuller(X, regression="c", autolag="AIC")  #
        df["rtn"].plot(figsize=(15,5))
        #plt.show()
        if result[0] < result[4]["1%"]:
            #null hypothesis is series is not stationary
            print("{0} {1} - {2}".format(index, self.start_date, self.end_date))
            print("ADF Statistic: %f" % result[0])
            print("p-value: %f" % result[1])
            print("Critical Values:")
            for key, value in result[4].items():
                print("\t%s: %.3f" % (key, value))
        #if x is white noise(p > 0.05)
        algo.timeseries.arima().ljungbox_test(X ** 2)
    
        #pacf to get p
        fig = plt.figure(figsize=(40,5))
        ax1=fig.add_subplot(111)
        fig = smt.graphics.plot_pacf(X,lags = 40,ax=ax1)      
        plt.show()   
        AR_lags = 13
        
        #AR model, change the lag
        order = (AR_lags,0)
        model = sm.tsa.ARMA(X,order).fit()   
        
        at = X - model.fittedvalues
        at2 = np.square(at)
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(at,label = 'at')
        plt.legend()
        plt.subplot(212)
        plt.plot(at2,label='at^2')
        plt.legend(loc=0)
        plt.show()
        
        algo.timeseries.arima().ljungbox_test(at2)
        #Ljung-Box test on at2 series
        #if p < 0.05 , is autocolrel, has ARCH
        #m = 25 # 我们检验25个自相关系数
        #acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
        #out = np.c_[range(1,26), acf[1:], q, p]
        #output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
        #output = output.set_index('lag')
        #print (f'LB test{output}')
    
        #Get value of p
        fig = plt.figure(figsize=(20,5))
        ax1=fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_pacf(at2,lags = 30,ax=ax1)
        plt.show()
        
        #returns = 100 * df['close'].pct_change().dropna()
        #returns.name = 'asset_returns'
        ##print(f'Average return: {round(returns.mean(), 2)}%')
        #returns.plot(title=f'{index} returns: {self.start_date} - {self.end_date}');
        
        #plt.tight_layout()
        # plt.savefig('images/ch5_im1.png')
        #plt.show()
        returns = 100 * df['rtn']
        print(f'Average return: {round(returns.mean(), 2)}%')
        training = returns.head(-10)
        test = returns.tail(10)
        model = arch_model(training, mean="AR", dist='Normal', vol='ARCH',  lags=AR_lags, p=7)
        model_fitted = model.fit(disp='off')
        print(model_fitted.summary())    
        print(model_fitted.params)    
        #Check if model's squared residuals has no correlation.
        algo.timeseries.arima().ljungbox_test(model_fitted.std_resid.dropna() ** 2)
        model_fitted.plot(annualize='D')
        plt.tight_layout()
        # plt.savefig('images/ch5_im3.png')
        plt.show()        
        
        model_fitted.hedgehog_plot()
        plt.show() 
        
        last_index = training.index[-1]
        #resid = model_fitted.resid[-AR_lags:]
        #a = np.array(model_fitted.params[1:9])
        #w = a[::-1] # 系数
        #for i in range(AR_lags):
        #    new = test[i] - (model_fitted.params[0] + w.dot(resid[-AR_lags:]))
        #    resid = np.append(resid,new)
        #print(len(resid))
        #for i in range(3):
        #    r_pred = model_fitted.params[0] + w.dot(resid[-AR_lags:])
        #at_pre = resid[-AR_lags:]
        #at_pre2 = at_pre**2
        #at_pre2
        
        forecast_result = model_fitted.forecast(horizon=10,start=last_index)
        pre = forecast_result.residual_variance.tail(1).T
        pre.index = test.index
        zero = pd.DataFrame(np.zeros(10), index=test.index)
        plt.figure(figsize=(10,4))
        plt.plot(test,label='realValue')
        plt.plot(pre,label='predictValue')
        plt.plot(zero,label='zero')
        plt.legend(loc=0)
        plt.show()
        
        
    def garch(self, index="GLD"):
        lags = 5
        self.start_date = "2019-01-03"
        self.end_date = TODAY
        if isChinaMkt(index):
            self.df = DailyPrice().load(security=index)
        else:
            self.df = USDailyPrice().load(security=index)
            
        df = self.df.loc[
            (self.df.index > self.start_date)
            & (self.df.index <= self.end_date)
        ]
        
        df['rtn'] = df['close'].pct_change()
        df = df[['close', 'rtn']].dropna(how = 'any')   
        returns = 100 * df['rtn']
        train = returns.head(-5)
        test = returns.tail(5)
        am = arch_model(train,mean='AR',lags=lags,vol='GARCH') 
        model_fitted = am.fit()        
        print (model_fitted.summary())
        print (model_fitted.params)
        model_fitted.plot()
        plt.plot(returns)        
        plt.show()
        #model_fitted.hedgehog_plot()
        #plt.show() 
        
        #Calculate volatility
        ini = model_fitted.resid[-lags:]
        a = np.array(model_fitted.params[1:lags+1])
        w = a[::-1] # 系数
        for i in range(lags):
            new = test[i] - (model_fitted.params[0] + w.dot(ini[-lags:]))
            ini = np.append(ini,new)
        print (len(ini))
        at_pre = ini[-(lags+2):]
        at_pre2 = at_pre**2

        ini2 = model_fitted.conditional_volatility[-2:] #上两个条件异方差值
        
        for i in range(lags):
            new = 0.027 + 0.123*at_pre2[i] + 0.8455*ini2[-1]
            ini2 = np.append(ini2,new)
        vol_pre = ini2[-lags:]
        vol_pre_df = pd.DataFrame(vol_pre, index=test.index)       
        
        plt.figure(figsize=(15,5))
        plt.plot(returns,label='origin_data')
        plt.plot(model_fitted.conditional_volatility,label='conditional_volatility')
        #x=range(479,489)
        plt.plot(vol_pre_df,'.r',label='predict_volatility')
        plt.legend(loc=0)   
        plt.show()