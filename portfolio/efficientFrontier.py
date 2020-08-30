# encoding: UTF-8


import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd

import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
from price import DailyPrice, Valuation, Security
from USprice import *
from common import *

class efficientFrontier(object):
    def __init__(self):
        self.returns_df = pd.DataFrame()
        self.weight_df = pd.DataFrame()
        self.security_list = ["GLD","TLT","SPY"]
        self.start_date = "2015-01-01"
        self.end_date = TODAY        
        
    def run(self):
        N_PORTFOLIOS = 10 ** 5
        N_DAYS = 252
        n_assets = len(self.security_list)
        prices_df = pd.DataFrame()
        for s in self.security_list:
            if isChinaMkt(s):
                df = DailyPrice().load(security=s)
            else:
                df = USDailyPrice().load(security=s)
                
            df = df.close.to_frame().loc[
                (df.index > self.start_date)
                & (df.index <= self.end_date)
            ]
            prices_df = pd.concat([prices_df, df], axis=1, sort=False, join="outer")
        prices_df = prices_df.dropna(how = 'any')
        prices_df.columns = self.security_list
        returns_df = prices_df.pct_change().dropna()
        avg_returns = returns_df.mean() * N_DAYS
        cov_mat = returns_df.cov() * N_DAYS   
        print(f'cov: {cov_mat}')
        
        np.random.seed(42)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))       
        weights /=  np.sum(weights, axis=1)[:, np.newaxis]
        portf_rtns = np.dot(weights, avg_returns)
        
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T, 
                                            np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)  
        portf_sharpe_ratio = portf_rtns / portf_vol
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                         'volatility': portf_vol,
                                         'sharpe_ratio': portf_sharpe_ratio})    

        N_POINTS = 100
        portf_vol_ef = []
        indices_to_skip = []
        
        portf_rtns_ef = np.linspace(portf_results_df.returns.min(), 
                                    portf_results_df.returns.max(), 
                                    N_POINTS)
        portf_rtns_ef = np.round(portf_rtns_ef, 3)    
        portf_rtns = np.round(portf_rtns, 3)
        
        for point_index in range(N_POINTS):
            if portf_rtns_ef[point_index] not in portf_rtns:
                indices_to_skip.append(point_index)
                continue
            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
            portf_vol_ef.append(np.min(portf_vol[matched_ind]))
        
        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)
        
        self.draw(portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights)
        
        
    def draw(self, portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights):
        MARKS = ['o', 'X', 'd', '*']
        
        fig, ax = plt.subplots()
        portf_results_df.plot(kind='scatter', x='volatility', 
                              y='returns', c='sharpe_ratio',
                              cmap='RdYlGn', edgecolors='black', 
                              ax=ax)
        ax.set(xlabel='Volatility', 
               ylabel='Expected Returns', 
               title='Efficient Frontier')
        ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
        
        for asset_index in range(len(self.security_list)):
            ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]), 
                        y=avg_returns[asset_index], 
                        marker=MARKS[asset_index], 
                        s=150, 
                        color='black',
                        label=self.security_list[asset_index])
        ax.legend()
        
        plt.tight_layout()
        #plt.savefig('images/ch7_im8.png')
        max_sharpe_portf = self.getMaxSharpePortfolio(portf_results_df, weights)
        min_vol_portf = self.getMinVolPortfolio(portf_results_df, weights)

        portf_results_df.plot(kind='scatter', x='volatility', 
                              y='returns', c='sharpe_ratio',
                              cmap='RdYlGn', edgecolors='black', 
                              ax=ax)
        ax.scatter(x=max_sharpe_portf.volatility, 
                   y=max_sharpe_portf.returns, 
                   c='black', marker='*', 
                   s=200, label='Max Sharpe Ratio')
        ax.scatter(x=min_vol_portf.volatility, 
                   y=min_vol_portf.returns, 
                   c='black', marker='P', 
                   s=200, label='Minimum Volatility')
        ax.set(xlabel='Volatility', ylabel='Expected Returns', 
               title='Efficient Frontier')
        ax.legend()
        
        plt.tight_layout()
        #plt.savefig('images/ch7_im11.png')
        plt.show()        
    
    def getMaxSharpePortfolio(self, portf_results_df, weights):
        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
        print('Maximum Sharpe Ratio portfolio ----')
        print('Performance')
        for index, value in max_sharpe_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.security_list, weights[np.argmax(portf_results_df.sharpe_ratio)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)        
        return max_sharpe_portf
     
    def getMinVolPortfolio(self, portf_results_df, weights):
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]          
        print('Minimum Volatility portfolio ----')
        print('Performance')
        for index, value in min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(self.security_list, weights[np.argmin(portf_results_df.volatility)]):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)
        return min_vol_portf