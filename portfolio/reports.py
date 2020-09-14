# encoding: UTF-8


import matplotlib.pyplot as plt
import warnings
import seaborn as sns

plt.style.use('seaborn')
sns.set_palette('cubehelix')
#plt.style.use('seaborn-colorblind') #alternative
#plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 100
warnings.simplefilter(action='ignore', category=FutureWarning)
import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
from price import DailyPrice, Valuation, Security
from USprice import *
from common import *


class portfolio(object):
    def __init__(self):
        self.returns_df = pd.DataFrame()
        self.weight_df = pd.DataFrame()
        self.security = "000001.XSHG"
        self.start_date = "2015-01-01"
        self.end_date = TODAY
        
    def run(self):
        if isChinaMkt(self.security):
            self.df = DailyPrice().load(security=self.security)
        else:
            self.df = USDailyPrice().load(security=self.security)
            
        self.df = self.df.close.to_frame().loc[
            (self.df.index > self.start_date)
            & (self.df.index <= self.end_date)
        ]      
        returns = self.df.pct_change().dropna()
        #1/n weight
        one_weight = len(self.df.columns) * [1 / len(self.df.columns)]
        weights=pd.DataFrame(
            np.repeat([one_weight],len(self.df)-1,axis=0), 
            index = returns.index
        )
        self.load(returns, weights)
        self.report()
        
    def load(self, returns, weights):
        self.returns_df  = returns
        self.weights_df = weights
        
    def report(self):
        portfolio_returns = np.multiply(self.weights_df, self.returns_df).agg(['sum'],axis=1)['sum']
        #pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)
        fig = pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)
        fig.savefig('returns_tear_sheet.pdf')
        #plt.show()
        
        
