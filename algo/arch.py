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
        self.start_date = "2016-01-03"
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
        plt.show()
        if result[0] < result[4]["1%"]:
            #null hypothesis is series is not stationary
            print("{0} {1} - {2}".format(index, self.start_date, self.end_date))
            print("ADF Statistic: %f" % result[0])
            print("p-value: %f" % result[1])
            print("Critical Values:")
            for key, value in result[4].items():
                print("\t%s: %.3f" % (key, value))
    
    
        #pacf to get p
        fig = plt.figure(figsize=(20,5))
        ax1=fig.add_subplot(111)
        fig = smt.graphics.plot_pacf(X,lags = 20,ax=ax1)      
        plt.show()   
        
        #AR model, change the lag
        order = (13,0)
        model = sm.tsa.ARMA(X,order).fit()   
        
        at = X -  model.fittedvalues
        at2 = np.square(at)
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(at,label = 'at')
        plt.legend()
        plt.subplot(212)
        plt.plot(at2,label='at^2')
        plt.legend(loc=0)
        plt.show()
        
        #Ljung-Box test to get coefficient
        #if p < 0.05 , is autocolrel, has ARCH
        m = 25 # 我们检验25个自相关系数
        acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
        out = np.c_[range(1,26), acf[1:], q, p]
        output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
        output = output.set_index('lag')
        print (f'LB test{output}')
    
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
        model = arch_model(training, dist='Normal', vol='GARCH',  p=1, o=0, q=1)
        model_fitted = model.fit(disp='off')
        print(model_fitted.summary())        
        model_fitted.plot(annualize='D')
        plt.tight_layout()
        # plt.savefig('images/ch5_im3.png')
        plt.show()        
        
        model_fitted.hedgehog_plot()
        plt.show() 
        
        
        pre = model_fitted.forecast(horizon=10,start=len(training)-1).iloc[len(training)-1]
        plt.figure(figsize=(10,4))
        plt.plot(test,label='realValue')
        pre.plot(label='predictValue')
        plt.plot(np.zeros(10),label='zero')
        plt.legend(loc=0)