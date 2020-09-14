[1mdiff --git a/indicators.py b/indicators.py[m
[1mindex 805b011..1fb2783 100644[m
[1m--- a/indicators.py[m
[1m+++ b/indicators.py[m
[36m@@ -11,6 +11,9 @@[m [mimport pandas as pd[m
 import _pickle as pickle[m
 [m
 [m
[32m+[m[41m        [m
[32m+[m[41m[m
[32m+[m[41m[m
 class IsST(SecurityBase):[m
     def __init__(self):[m
         super(IsST, self).__init__("IsST")[m
[36m@@ -46,7 +49,6 @@[m [mclass MoneyFlow(SecurityBase):[m
         df = get_money_flow(index, start_date, end_date)[m
         return df[m
 [m
[31m-[m
 class indicators(object):[m
     def __init__(self, class_name="DailyPrice", db_name="DailyIndicators"):[m
         # Load price timeseries[m
[36m@@ -151,4 +153,4 @@[m [mclass indicators(object):[m
                 self.df_dict[security].index = self.df_dict[security]["index"][m
             fp = open(self.pickle_file, "wb")[m
             pickle.dump(self.df_dict, fp)[m
[31m-        return self.df_dict[m
[32m+[m[32m        return self.df_dict[m
\ No newline at end of file[m
[1mdiff --git a/jqdata-download.py b/jqdata-download.py[m
[1mindex 6782803..6a33297 100644[m
[1m--- a/jqdata-download.py[m
[1m+++ b/jqdata-download.py[m
[36m@@ -718,6 +718,10 @@[m [mdef updateAllUS():[m
     USSecurity().update()[m
     USDailyPrice().updateAll()[m
     [m
[32m+[m[32mdef updateAllFinance():[m
[32m+[m[32m    FundamentalQuarter().updateAll()[m
[32m+[m[32m    IncomeQuarter().updateAll()[m
[32m+[m[32m    BalanceQuarter().updateAll()[m
 [m
 def test():[m
     q = ([m
[36m@@ -780,8 +784,9 @@[m [mif __name__ == "__main__":[m
         "garch": algo.arch.arch().garch,[m
         "arima": algo.timeseries.arima().test,[m
         "portfolio": portfolio.reports.portfolio().run,[m
[31m-        "ef": portfolio.efficientFrontier.efficientFrontier().run,[m
[31m-        [m
[32m+[m[32m        "ef": portfolio.efficientFrontier.efficientFrontier().run_montecarlo,[m
[32m+[m[32m        "ef_volopt": portfolio.efficientFrontier.efficientFrontier().run_vol_optimization,[m
[32m+[m[32m        "ef_sharpeopt": portfolio.efficientFrontier.efficientFrontier().run_sharpe_optimization,[m
         [m
         "updateIndicatorsD": indicators().update,[m
         "updateIndicatorsW": indicators([m
[36m@@ -804,7 +809,8 @@[m [mif __name__ == "__main__":[m
         "updateAll": updateAll,[m
         "updateAllUS": updateAllUS,[m
         "updateIndustryPriceD": IndustryDailyPrice().updateAll,[m
[31m-        "updateConcept": Concept().updateAll[m
[32m+[m[32m        "updateConcept": Concept().updateAll,[m
[32m+[m[32m        "updateAllFinance": updateAllFinance[m
     }[m
     parser = argparse.ArgumentParser()[m
     parser.add_argument("command", choices=FUNCTION_MAP.keys())[m
[1mdiff --git a/portfolio/efficientFrontier.py b/portfolio/efficientFrontier.py[m
[1mindex 8922610..d61a504 100644[m
[1m--- a/portfolio/efficientFrontier.py[m
[1m+++ b/portfolio/efficientFrontier.py[m
[36m@@ -1,154 +1,279 @@[m
[31m-# encoding: UTF-8[m
[31m-[m
[31m-[m
[31m-import matplotlib.pyplot as plt[m
[31m-import warnings[m
[31m-import seaborn as sns[m
[31m-import pandas as pd[m
[31m-[m
[31m-import yfinance as yf[m
[31m-import numpy as np[m
[31m-import pandas as pd[m
[31m-import pyfolio as pf[m
[31m-from price import DailyPrice, Valuation, Security[m
[31m-from USprice import *[m
[31m-from common import *[m
[31m-[m
[31m-class efficientFrontier(object):[m
[31m-    def __init__(self):[m
[31m-        self.returns_df = pd.DataFrame()[m
[31m-        self.weight_df = pd.DataFrame()[m
[31m-        self.security_list = ["GLD","TLT","SPY"][m
[31m-        self.start_date = "2020-03-19"[m
[31m-        self.end_date = TODAY        [m
[31m-        [m
[31m-    def run(self):[m
[31m-        N_PORTFOLIOS = 10 ** 5[m
[31m-        N_DAYS = 252[m
[31m-        n_assets = len(self.security_list)[m
[31m-        prices_df = pd.DataFrame()[m
[31m-        for s in self.security_list:[m
[31m-            if isChinaMkt(s):[m
[31m-                df = DailyPrice().load(security=s)[m
[31m-            else:[m
[31m-                df = USDailyPrice().load(security=s)[m
[31m-                [m
[31m-            df = df.close.to_frame().loc[[m
[31m-                (df.index > self.start_date)[m
[31m-                & (df.index <= self.end_date)[m
[31m-            ][m
[31m-            prices_df = pd.concat([prices_df, df], axis=1, sort=False, join="outer")[m
[31m-        prices_df = prices_df.dropna(how = 'any')[m
[31m-        prices_df.columns = self.security_list[m
[31m-        returns_df = prices_df.pct_change().dropna()[m
[31m-        avg_returns = returns_df.mean() * N_DAYS[m
[31m-        cov_mat = returns_df.cov() * N_DAYS   [m
[31m-        print(f'cov: {cov_mat}')[m
[31m-        [m
[31m-        np.random.seed(42)[m
[31m-        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))       [m
[31m-        weights /=  np.sum(weights, axis=1)[:, np.newaxis][m
[31m-        portf_rtns = np.dot(weights, avg_returns)[m
[31m-        [m
[31m-        portf_vol = [][m
[31m-        for i in range(0, len(weights)):[m
[31m-            portf_vol.append(np.sqrt(np.dot(weights[i].T, [m
[31m-                                            np.dot(cov_mat, weights[i]))))[m
[31m-        portf_vol = np.array(portf_vol)  [m
[31m-        portf_sharpe_ratio = portf_rtns / portf_vol[m
[31m-        portf_results_df = pd.DataFrame({'returns': portf_rtns,[m
[31m-                                         'volatility': portf_vol,[m
[31m-                                         'sharpe_ratio': portf_sharpe_ratio})    [m
[31m-[m
[31m-        N_POINTS = 100[m
[31m-        portf_vol_ef = [][m
[31m-        indices_to_skip = [][m
[31m-        [m
[31m-        portf_rtns_ef = np.linspace(portf_results_df.returns.min(), [m
[31m-                                    portf_results_df.returns.max(), [m
[31m-                                    N_POINTS)[m
[31m-        portf_rtns_ef = np.round(portf_rtns_ef, 3)    [m
[31m-        portf_rtns = np.round(portf_rtns, 3)[m
[31m-        [m
[31m-        for point_index in range(N_POINTS):[m
[31m-            if portf_rtns_ef[point_index] not in portf_rtns:[m
[31m-                indices_to_skip.append(point_index)[m
[31m-                continue[m
[31m-            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])[m
[31m-            portf_vol_ef.append(np.min(portf_vol[matched_ind]))[m
[31m-        [m
[31m-        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)[m
[31m-        [m
[31m-        self.draw(portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights)[m
[31m-        [m
[31m-        [m
[31m-    def draw(self, portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights):[m
[31m-        MARKS = ['o', 'X', 'd', '*'][m
[31m-        [m
[31m-        fig, ax = plt.subplots()[m
[31m-        portf_results_df.plot(kind='scatter', x='volatility', [m
[31m-                              y='returns', c='sharpe_ratio',[m
[31m-                              cmap='RdYlGn', edgecolors='black', [m
[31m-                              ax=ax)[m
[31m-        ax.set(xlabel='Volatility', [m
[31m-               ylabel='Expected Returns', [m
[31m-               title='Efficient Frontier')[m
[31m-        ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')[m
[31m-        [m
[31m-        for asset_index in range(len(self.security_list)):[m
[31m-            ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]), [m
[31m-                        y=avg_returns[asset_index], [m
[31m-                        marker=MARKS[asset_index], [m
[31m-                        s=150, [m
[31m-                        color='black',[m
[31m-                        label=self.security_list[asset_index])[m
[31m-        ax.legend()[m
[31m-        [m
[31m-        plt.tight_layout()[m
[31m-        #plt.savefig('images/ch7_im8.png')[m
[31m-        max_sharpe_portf = self.getMaxSharpePortfolio(portf_results_df, weights)[m
[31m-        min_vol_portf = self.getMinVolPortfolio(portf_results_df, weights)[m
[31m-[m
[31m-        portf_results_df.plot(kind='scatter', x='volatility', [m
[31m-                              y='returns', c='sharpe_ratio',[m
[31m-                              cmap='RdYlGn', edgecolors='black', [m
[31m-                              ax=ax)[m
[31m-        ax.scatter(x=max_sharpe_portf.volatility, [m
[31m-                   y=max_sharpe_portf.returns, [m
[31m-                   c='black', marker='*', [m
[31m-                   s=200, label='Max Sharpe Ratio')[m
[31m-        ax.scatter(x=min_vol_portf.volatility, [m
[31m-                   y=min_vol_portf.returns, [m
[31m-                   c='black', marker='P', [m
[31m-                   s=200, label='Minimum Volatility')[m
[31m-        ax.set(xlabel='Volatility', ylabel='Expected Returns', [m
[31m-               title='Efficient Frontier')[m
[31m-        ax.legend()[m
[31m-        [m
[31m-        plt.tight_layout()[m
[31m-        #plt.savefig('images/ch7_im11.png')[m
[31m-        plt.show()        [m
[31m-    [m
[31m-    def getMaxSharpePortfolio(self, portf_results_df, weights):[m
[31m-        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)[m
[31m-        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind][m
[31m-        print('Maximum Sharpe Ratio portfolio ----')[m
[31m-        print('Performance')[m
[31m-        for index, value in max_sharpe_portf.items():[m
[31m-            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m
[31m-        print('\nWeights')[m
[31m-        for x, y in zip(self.security_list, weights[np.argmax(portf_results_df.sharpe_ratio)]):[m
[31m-            print(f'{x}: {100*y:.2f}% ', end="", flush=True)        [m
[31m-        return max_sharpe_portf[m
[31m-     [m
[31m-    def getMinVolPortfolio(self, portf_results_df, weights):[m
[31m-        min_vol_ind = np.argmin(portf_results_df.volatility)[m
[31m-        min_vol_portf = portf_results_df.loc[min_vol_ind]          [m
[31m-        print('Minimum Volatility portfolio ----')[m
[31m-        print('Performance')[m
[31m-        for index, value in min_vol_portf.items():[m
[31m-            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m
[31m-        print('\nWeights')[m
[31m-        for x, y in zip(self.security_list, weights[np.argmin(portf_results_df.volatility)]):[m
[31m-            print(f'{x}: {100*y:.2f}% ', end="", flush=True)[m
[32m+[m[32m# encoding: UTF-8[m[41m[m
[32m+[m[41m[m
[32m+[m[41m[m
[32m+[m[32mimport matplotlib.pyplot as plt[m[41m[m
[32m+[m[32mimport warnings[m[41m[m
[32m+[m[32mimport seaborn as sns[m[41m[m
[32m+[m[32mimport pandas as pd[m[41m[m
[32m+[m[32mimport scipy.optimize as sco[m[41m[m
[32m+[m[41m[m
[32m+[m[32mimport yfinance as yf[m[41m[m
[32m+[m[32mimport numpy as np[m[41m[m
[32m+[m[32mimport pandas as pd[m[41m[m
[32m+[m[32mimport pyfolio as pf[m[41m[m
[32m+[m[32mfrom price import DailyPrice, Valuation, Security[m[41m[m
[32m+[m[32mfrom USprice import *[m[41m[m
[32m+[m[32mfrom common import *[m[41m[m
[32m+[m[41m[m
[32m+[m[32mN_PORTFOLIOS = 10 ** 5[m[41m[m
[32m+[m[32mN_DAYS = 252[m[41m[m
[32m+[m[41m[m
[32m+[m[32mclass efficientFrontier(object):[m[41m[m
[32m+[m[32m    def __init__(self):[m[41m[m
[32m+[m[32m        self.returns_df = pd.DataFrame()[m[41m[m
[32m+[m[32m        self.weight_df = pd.DataFrame()[m[41m[m
[32m+[m[32m        self.security_list = ["GLD","TLT","SPY"][m[41m[m
[32m+[m[32m        self.start_date = "2020-03-19"[m[41m[m
[32m+[m[32m        self.end_date = TODAY[m[41m   [m
[32m+[m[41m        [m
[32m+[m[32m    def load(self):[m[41m[m
[32m+[m[41m[m
[32m+[m[32m        prices_df = pd.DataFrame()[m[41m[m
[32m+[m[32m        for s in self.security_list:[m[41m[m
[32m+[m[32m            if isChinaMkt(s):[m[41m[m
[32m+[m[32m                df = DailyPrice().load(security=s)[m[41m[m
[32m+[m[32m            else:[m[41m[m
[32m+[m[32m                df = USDailyPrice().load(security=s)[m[41m[m
[32m+[m[41m                [m
[32m+[m[32m            df = df.close.to_frame().loc[[m[41m[m
[32m+[m[32m                (df.index > self.start_date)[m[41m[m
[32m+[m[32m                & (df.index <= self.end_date)[m[41m[m
[32m+[m[32m            ][m[41m[m
[32m+[m[32m            prices_df = pd.concat([prices_df, df], axis=1, sort=False, join="outer")[m[41m[m
[32m+[m[32m        prices_df = prices_df.dropna(how = 'any')[m[41m[m
[32m+[m[32m        prices_df.columns = self.security_list[m[41m[m
[32m+[m[32m        returns_df = prices_df.pct_change().dropna()[m[41m[m
[32m+[m[32m        return returns_df[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m    def run_montecarlo(self):[m[41m[m
[32m+[m[41m[m
[32m+[m[32m        n_assets = len(self.security_list)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        returns_df = self.load()[m[41m[m
[32m+[m[32m        avg_returns = returns_df.mean() * N_DAYS[m[41m[m
[32m+[m[32m        cov_mat = returns_df.cov() * N_DAYS[m[41m   [m
[32m+[m[32m        print(f'cov: {cov_mat}')[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        np.random.seed(42)[m[41m[m
[32m+[m[32m        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))[m[41m       [m
[32m+[m[32m        weights /=  np.sum(weights, axis=1)[:, np.newaxis][m[41m[m
[32m+[m[32m        portf_rtns = np.dot(weights, avg_returns)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        portf_vol = [][m[41m[m
[32m+[m[32m        for i in range(0, len(weights)):[m[41m[m
[32m+[m[32m            portf_vol.append(np.sqrt(np.dot(weights[i].T,[m[41m [m
[32m+[m[32m                                            np.dot(cov_mat, weights[i]))))[m[41m[m
[32m+[m[32m        portf_vol = np.array(portf_vol)[m[41m  [m
[32m+[m[32m        portf_sharpe_ratio = portf_rtns / portf_vol[m[41m[m
[32m+[m[32m        portf_results_df = pd.DataFrame({'returns': portf_rtns,[m[41m[m
[32m+[m[32m                                         'volatility': portf_vol,[m[41m[m
[32m+[m[32m                                         'sharpe_ratio': portf_sharpe_ratio})[m[41m    [m
[32m+[m[41m[m
[32m+[m[32m        N_POINTS = 100[m[41m[m
[32m+[m[32m        portf_vol_ef = [][m[41m[m
[32m+[m[32m        indices_to_skip = [][m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        portf_rtns_ef = np.linspace(portf_results_df.returns.min(),[m[41m [m
[32m+[m[32m                                    portf_results_df.returns.max(),[m[41m [m
[32m+[m[32m                                    N_POINTS)[m[41m[m
[32m+[m[32m        portf_rtns_ef = np.round(portf_rtns_ef, 3)[m[41m    [m
[32m+[m[32m        portf_rtns = np.round(portf_rtns, 3)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        for point_index in range(N_POINTS):[m[41m[m
[32m+[m[32m            if portf_rtns_ef[point_index] not in portf_rtns:[m[41m[m
[32m+[m[32m                indices_to_skip.append(point_index)[m[41m[m
[32m+[m[32m                continue[m[41m[m
[32m+[m[32m            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])[m[41m[m
[32m+[m[32m            portf_vol_ef.append(np.min(portf_vol[matched_ind]))[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        self.draw(portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights)[m[41m[m
[32m+[m[41m[m
[32m+[m[32m    def run_sharpe_optimization(self):[m[41m[m
[32m+[m[32m        def get_portf_rtn(w, avg_rtns):[m[41m[m
[32m+[m[32m            return np.sum(avg_rtns * w)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        def get_portf_vol(w, avg_rtns, cov_mat):[m[41m[m
[32m+[m[32m            return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):[m[41m[m
[32m+[m[32m            #objective function[m[41m[m
[32m+[m[32m            portf_returns = np.sum(avg_rtns * w)[m[41m[m
[32m+[m[32m            portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))[m[41m[m
[32m+[m[32m            portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility[m[41m[m
[32m+[m[32m            return -portf_sharpe_ratio[m[41m     [m
[32m+[m[41m        [m
[32m+[m[32m        returns_df = self.load()[m[41m[m
[32m+[m[32m        avg_returns = returns_df.mean() * N_DAYS[m[41m[m
[32m+[m[32m        cov_mat = returns_df.cov() * N_DAYS[m[41m   [m
[32m+[m[32m        n_assets = len(avg_returns)[m[41m[m
[32m+[m[32m        RF_RATE = 0[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        args = (avg_returns, cov_mat, RF_RATE)[m[41m[m
[32m+[m[32m        constraints = ({'type': 'eq',[m[41m [m
[32m+[m[32m                        'fun': lambda x: np.sum(x) - 1})[m[41m[m
[32m+[m[32m        bounds = tuple((0,1) for asset in range(n_assets))[m[41m[m
[32m+[m[32m        initial_guess = n_assets * [1. / n_assets][m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        max_sharpe_portf = sco.minimize(neg_sharpe_ratio,[m[41m [m
[32m+[m[32m                                        x0=initial_guess,[m[41m [m
[32m+[m[32m                                        args=args,[m[41m[m
[32m+[m[32m                                        method='SLSQP',[m[41m [m
[32m+[m[32m                                        bounds=bounds,[m[41m [m
[32m+[m[32m                                        constraints=constraints)[m[41m        [m
[32m+[m[41m        [m
[32m+[m[32m        max_sharpe_portf_w = max_sharpe_portf['x'][m[41m[m
[32m+[m[32m        max_sharpe_portf = {'Return': get_portf_rtn(max_sharpe_portf_w,[m[41m [m
[32m+[m[32m                                                    avg_returns),[m[41m[m
[32m+[m[32m                            'Volatility': get_portf_vol(max_sharpe_portf_w,[m[41m [m
[32m+[m[32m                                                        avg_returns,[m[41m [m
[32m+[m[32m                                                        cov_mat),[m[41m[m
[32m+[m[32m                            'Sharpe Ratio': -max_sharpe_portf['fun']}[m[41m[m
[32m+[m[32m        print('Maximum Sharpe Ratio portfolio ----')[m[41m[m
[32m+[m[32m        print('Performance')[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        for index, value in max_sharpe_portf.items():[m[41m[m
[32m+[m[32m            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        print('\nWeights')[m[41m[m
[32m+[m[32m        for x, y in zip(self.security_list, max_sharpe_portf_w):[m[41m[m
[32m+[m[32m            print(f'{x}: {100*y:.2f}% ', end="", flush=True)[m[41m        [m
[32m+[m[41m[m
[32m+[m[32m    def run_vol_optimization(self):[m[41m[m
[32m+[m[32m        def get_portf_rtn(w, avg_rtns):[m[41m[m
[32m+[m[32m            return np.sum(avg_rtns * w)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        def get_portf_vol(w, avg_rtns, cov_mat):[m[41m[m
[32m+[m[32m            return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))[m[41m        [m
[32m+[m[41m        [m
[32m+[m[32m        def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):[m[41m[m
[32m+[m[41m            [m
[32m+[m[32m            efficient_portfolios = [][m[41m[m
[32m+[m[41m            [m
[32m+[m[32m            n_assets = len(avg_returns)[m[41m[m
[32m+[m[32m            args = (avg_returns, cov_mat)[m[41m[m
[32m+[m[32m            bounds = tuple((0,1) for asset in range(n_assets))[m[41m[m
[32m+[m[32m            initial_guess = n_assets * [1. / n_assets, ][m[41m[m
[32m+[m[41m            [m
[32m+[m[32m            for ret in rtns_range:[m[41m[m
[32m+[m[32m                constraints = ({'type': 'eq',[m[41m [m
[32m+[m[32m                                'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret},[m[41m[m
[32m+[m[32m                               {'type': 'eq',[m[41m [m
[32m+[m[32m                                'fun': lambda x: np.sum(x) - 1})[m[41m[m
[32m+[m[32m                efficient_portfolio = sco.minimize(get_portf_vol, initial_guess,[m[41m [m
[32m+[m[32m                                                   args=args, method='SLSQP',[m[41m [m
[32m+[m[32m                                                   constraints=constraints,[m[41m[m
[32m+[m[32m                                                   bounds=bounds)[m[41m[m
[32m+[m[32m                efficient_portfolios.append(efficient_portfolio)[m[41m[m
[32m+[m[41m            [m
[32m+[m[32m            return efficient_portfolios[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        returns_df = self.load()[m[41m[m
[32m+[m[32m        avg_returns = returns_df.mean() * N_DAYS[m[41m[m
[32m+[m[32m        cov_mat = returns_df.cov() * N_DAYS[m[41m   [m
[32m+[m[32m        rtns_range = np.linspace(-0.1, 2, 200)[m[41m[m
[32m+[m[32m        efficient_portfolios = get_efficient_frontier(avg_returns,[m[41m [m
[32m+[m[32m                                                      cov_mat,[m[41m [m
[32m+[m[32m                                                      rtns_range)[m[41m[m
[32m+[m[32m        vols_range = [x['fun'] for x in efficient_portfolios][m[41m[m
[32m+[m[32m        fig, ax = plt.subplots()[m[41m[m
[32m+[m[32m        ax.plot(vols_range, rtns_range, 'b--', linewidth=3)[m[41m[m
[32m+[m[32m        ax.set(xlabel='Volatility',[m[41m [m
[32m+[m[32m               ylabel='Expected Returns',[m[41m [m
[32m+[m[32m               title='Efficient Frontier')[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        plt.tight_layout()[m[41m[m
[32m+[m[32m        #plt.savefig('images/ch7_im12.png')[m[41m[m
[32m+[m[32m        plt.show()[m[41m     [m
[32m+[m[41m[m
[32m+[m[32m        min_vol_ind = np.argmin(vols_range)[m[41m[m
[32m+[m[32m        min_vol_portf_rtn = rtns_range[min_vol_ind][m[41m[m
[32m+[m[32m        min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun'][m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        min_vol_portf = {'Return': min_vol_portf_rtn,[m[41m[m
[32m+[m[32m                         'Volatility': min_vol_portf_vol,[m[41m[m
[32m+[m[32m                         'Sharpe Ratio': (min_vol_portf_rtn /[m[41m [m
[32m+[m[32m                                          min_vol_portf_vol)}[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        print('Minimum Volatility portfolio ----')[m[41m[m
[32m+[m[32m        print('Performance')[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        for index, value in min_vol_portf.items():[m[41m[m
[32m+[m[32m            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        print('\nWeights')[m[41m[m
[32m+[m[32m        for x, y in zip(self.security_list, efficient_portfolios[min_vol_ind]['x']):[m[41m[m
[32m+[m[32m            print(f'{x}: {100*y:.2f}% ', end="", flush=True)[m[41m        [m
[32m+[m[41m        [m
[32m+[m[41m        [m
[32m+[m[32m    def draw(self, portf_results_df, portf_rtns_ef, portf_vol_ef, cov_mat, avg_returns, weights):[m[41m[m
[32m+[m[32m        MARKS = ['o', 'X', 'd', '*'][m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        fig, ax = plt.subplots()[m[41m[m
[32m+[m[32m        portf_results_df.plot(kind='scatter', x='volatility',[m[41m [m
[32m+[m[32m                              y='returns', c='sharpe_ratio',[m[41m[m
[32m+[m[32m                              cmap='RdYlGn', edgecolors='black',[m[41m [m
[32m+[m[32m                              ax=ax)[m[41m[m
[32m+[m[32m        ax.set(xlabel='Volatility',[m[41m [m
[32m+[m[32m               ylabel='Expected Returns',[m[41m [m
[32m+[m[32m               title='Efficient Frontier')[m[41m[m
[32m+[m[32m        ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        for asset_index in range(len(self.security_list)):[m[41m[m
[32m+[m[32m            ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),[m[41m [m
[32m+[m[32m                        y=avg_returns[asset_index],[m[41m [m
[32m+[m[32m                        marker=MARKS[asset_index],[m[41m [m
[32m+[m[32m                        s=150,[m[41m [m
[32m+[m[32m                        color='black',[m[41m[m
[32m+[m[32m                        label=self.security_list[asset_index])[m[41m[m
[32m+[m[32m        ax.legend()[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        plt.tight_layout()[m[41m[m
[32m+[m[32m        #plt.savefig('images/ch7_im8.png')[m[41m[m
[32m+[m[32m        max_sharpe_portf = self.getMaxSharpePortfolio(portf_results_df, weights)[m[41m[m
[32m+[m[32m        min_vol_portf = self.getMinVolPortfolio(portf_results_df, weights)[m[41m[m
[32m+[m[41m[m
[32m+[m[32m        portf_results_df.plot(kind='scatter', x='volatility',[m[41m [m
[32m+[m[32m                              y='returns', c='sharpe_ratio',[m[41m[m
[32m+[m[32m                              cmap='RdYlGn', edgecolors='black',[m[41m [m
[32m+[m[32m                              ax=ax)[m[41m[m
[32m+[m[32m        ax.scatter(x=max_sharpe_portf.volatility,[m[41m [m
[32m+[m[32m                   y=max_sharpe_portf.returns,[m[41m [m
[32m+[m[32m                   c='black', marker='*',[m[41m [m
[32m+[m[32m                   s=200, label='Max Sharpe Ratio')[m[41m[m
[32m+[m[32m        ax.scatter(x=min_vol_portf.volatility,[m[41m [m
[32m+[m[32m                   y=min_vol_portf.returns,[m[41m [m
[32m+[m[32m                   c='black', marker='P',[m[41m [m
[32m+[m[32m                   s=200, label='Minimum Volatility')[m[41m[m
[32m+[m[32m        ax.set(xlabel='Volatility', ylabel='Expected Returns',[m[41m [m
[32m+[m[32m               title='Efficient Frontier')[m[41m[m
[32m+[m[32m        ax.legend()[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m        plt.tight_layout()[m[41m[m
[32m+[m[32m        #plt.savefig('images/ch7_im11.png')[m[41m[m
[32m+[m[32m        plt.show()[m[41m        [m
[32m+[m[41m    [m
[32m+[m[32m    def getMaxSharpePortfolio(self, portf_results_df, weights):[m[41m[m
[32m+[m[32m        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)[m[41m[m
[32m+[m[32m        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind][m[41m[m
[32m+[m[32m        print('Maximum Sharpe Ratio portfolio ----')[m[41m[m
[32m+[m[32m        print('Performance')[m[41m[m
[32m+[m[32m        for index, value in max_sharpe_portf.items():[m[41m[m
[32m+[m[32m            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m[41m[m
[32m+[m[32m        print('\nWeights')[m[41m[m
[32m+[m[32m        for x, y in zip(self.security_list, weights[np.argmax(portf_results_df.sharpe_ratio)]):[m[41m[m
[32m+[m[32m            print(f'{x}: {100*y:.2f}% ', end="", flush=True)[m[41m        [m
[32m+[m[32m        return max_sharpe_portf[m[41m[m
[32m+[m[41m     [m
[32m+[m[32m    def getMinVolPortfolio(self, portf_results_df, weights):[m[41m[m
[32m+[m[32m        min_vol_ind = np.argmin(portf_results_df.volatility)[m[41m[m
[32m+[m[32m        min_vol_portf = portf_results_df.loc[min_vol_ind][m[41m          [m
[32m+[m[32m        print('Minimum Volatility portfolio ----')[m[41m[m
[32m+[m[32m        print('Performance')[m[41m[m
[32m+[m[32m        for index, value in min_vol_portf.items():[m[41m[m
[32m+[m[32m            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)[m[41m[m
[32m+[m[32m        print('\nWeights')[m[41m[m
[32m+[m[32m        for x, y in zip(self.security_list, weights[np.argmin(portf_results_df.volatility)]):[m[41m[m
[32m+[m[32m            print(f'{x}: {100*y:.2f}% ', end="", flush=True)[m[41m[m
         return min_vol_portf[m
\ No newline at end of file[m
[1mdiff --git a/portfolio/reports.py b/portfolio/reports.py[m
[1mindex 8aba8fe..eb4a8a9 100644[m
[1m--- a/portfolio/reports.py[m
[1m+++ b/portfolio/reports.py[m
[36m@@ -1,62 +1,62 @@[m
[31m-# encoding: UTF-8[m
[31m-[m
[31m-[m
[31m-import matplotlib.pyplot as plt[m
[31m-import warnings[m
[31m-import seaborn as sns[m
[31m-[m
[31m-plt.style.use('seaborn')[m
[31m-sns.set_palette('cubehelix')[m
[31m-#plt.style.use('seaborn-colorblind') #alternative[m
[31m-#plt.rcParams['figure.figsize'] = [8, 4.5][m
[31m-plt.rcParams['figure.dpi'] = 100[m
[31m-warnings.simplefilter(action='ignore', category=FutureWarning)[m
[31m-import yfinance as yf[m
[31m-import numpy as np[m
[31m-import pandas as pd[m
[31m-import pyfolio as pf[m
[31m-from price import DailyPrice, Valuation, Security[m
[31m-from USprice import *[m
[31m-from common import *[m
[31m-[m
[31m-[m
[31m-class portfolio(object):[m
[31m-    def __init__(self):[m
[31m-        self.returns_df = pd.DataFrame()[m
[31m-        self.weight_df = pd.DataFrame()[m
[31m-        self.security = "000001.XSHG"[m
[31m-        self.start_date = "2015-01-01"[m
[31m-        self.end_date = TODAY[m
[31m-        [m
[31m-    def run(self):[m
[31m-        if isChinaMkt(self.security):[m
[31m-            self.df = DailyPrice().load(security=self.security)[m
[31m-        else:[m
[31m-            self.df = USDailyPrice().load(security=self.security)[m
[31m-            [m
[31m-        self.df = self.df.close.to_frame().loc[[m
[31m-            (self.df.index > self.start_date)[m
[31m-            & (self.df.index <= self.end_date)[m
[31m-        ]      [m
[31m-        returns = self.df.pct_change().dropna()[m
[31m-        #1/n weight[m
[31m-        one_weight = len(self.df.columns) * [1 / len(self.df.columns)][m
[31m-        weights=pd.DataFrame([m
[31m-            np.repeat([one_weight],len(self.df)-1,axis=0), [m
[31m-            index = returns.index[m
[31m-        )[m
[31m-        self.load(returns, weights)[m
[31m-        self.report()[m
[31m-        [m
[31m-    def load(self, returns, weights):[m
[31m-        self.returns_df  = returns[m
[31m-        self.weights_df = weights[m
[31m-        [m
[31m-    def report(self):[m
[31m-        portfolio_returns = np.multiply(self.weights_df, self.returns_df).agg(['sum'],axis=1)['sum'][m
[31m-        #pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)[m
[31m-        fig = pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)[m
[31m-        fig.savefig('returns_tear_sheet.pdf')[m
[31m-        #plt.show()[m
[31m-        [m
[31m-        [m
[32m+[m[32m# encoding: UTF-8[m[41m[m
[32m+[m[41m[m
[32m+[m[41m[m
[32m+[m[32mimport matplotlib.pyplot as plt[m[41m[m
[32m+[m[32mimport warnings[m[41m[m
[32m+[m[32mimport seaborn as sns[m[41m[m
[32m+[m[41m[m
[32m+[m[32mplt.style.use('seaborn')[m[41m[m
[32m+[m[32msns.set_palette('cubehelix')[m[41m[m
[32m+[m[32m#plt.style.use('seaborn-colorblind') #alternative[m[41m[m
[32m+[m[32m#plt.rcParams['figure.figsize'] = [8, 4.5][m[41m[m
[32m+[m[32mplt.rcParams['figure.dpi'] = 100[m[41m[m
[32m+[m[32mwarnings.simplefilter(action='ignore', category=FutureWarning)[m[41m[m
[32m+[m[32mimport yfinance as yf[m[41m[m
[32m+[m[32mimport numpy as np[m[41m[m
[32m+[m[32mimport pandas as pd[m[41m[m
[32m+[m[32mimport pyfolio as pf[m[41m[m
[32m+[m[32mfrom price import DailyPrice, Valuation, Security[m[41m[m
[32m+[m[32mfrom USprice import *[m[41m[m
[32m+[m[32mfrom common import *[m[41m[m
[32m+[m[41m[m
[32m+[m[41m[m
[32m+[m[32mclass portfolio(object):[m[41m[m
[32m+[m[32m    def __init__(self):[m[41m[m
[32m+[m[32m        self.returns_df = pd.DataFrame()[m[41m[m
[32m+[m[32m        self.weight_df = pd.DataFrame()[m[41m[m
[32m+[m[32m        self.security = "000001.XSHG"[m[41m[m
[32m+[m[32m        self.start_date = "2015-01-01"[m[41m[m
[32m+[m[32m        self.end_date = TODAY[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m    def run(self):[m[41m[m
[32m+[m[32m        if isChinaMkt(self.security):[m[41m[m
[32m+[m[32m            self.df = DailyPrice().load(security=self.security)[m[41m[m
[32m+[m[32m        else:[m[41m[m
[32m+[m[32m            self.df = USDailyPrice().load(security=self.security)[m[41m[m
[32m+[m[41m            [m
[32m+[m[32m        self.df = self.df.close.to_frame().loc[[m[41m[m
[32m+[m[32m            (self.df.index > self.start_date)[m[41m[m
[32m+[m[32m            & (self.df.index <= self.end_date)[m[41m[m
[32m+[m[32m        ][m[41m      [m
[32m+[m[32m        returns = self.df.pct_change().dropna()[m[41m[m
[32m+[m[32m        #1/n weight[m[41m[m
[32m+[m[32m        one_weight = len(self.df.columns) * [1 / len(self.df.columns)][m[41m[m
[32m+[m[32m        weights=pd.DataFrame([m[41m[m
[32m+[m[32m            np.repeat([one_weight],len(self.df)-1,axis=0),[m[41m [m
[32m+[m[32m            index = returns.index[m[41m[m
[32m+[m[32m        )[m[41m[m
[32m+[m[32m        self.load(returns, weights)[m[41m[m
[32m+[m[32m        self.report()[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m    def load(self, returns, weights):[m[41m[m
[32m+[m[32m        self.returns_df  = returns[m[41m[m
[32m+[m[32m        self.weights_df = weights[m[41m[m
[32m+[m[41m        [m
[32m+[m[32m    def report(self):[m[41m[m
[32m+[m[32m        portfolio_returns = np.multiply(self.weights_df, self.returns_df).agg(['sum'],axis=1)['sum'][m[41m[m
[32m+[m[32m        #pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)[m[41m[m
[32m+[m[32m        fig = pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)[m[41m[m
[32m+[m[32m        fig.savefig('returns_tear_sheet.pdf')[m[41m[m
[32m+[m[32m        #plt.show()[m[41m[m
[32m+[m[41m        [m
[32m+[m[41m        [m
