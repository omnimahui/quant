# encoding: UTF-8

from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
import statsmodels.formula.api as smapi
from genhurst import genhurst
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
from scipy.stats.stats import pearsonr

import scipy.stats as scs
import seaborn as sns 
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose

import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode
import plotly.io as pio
import plotly
from fbprophet import Prophet

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)

class algo(object):
    def __init__(self):  
        self.start_date = "2018-01-01"
        self.end_date = TODAY
        self.training_start_date = "2016-01-27"
        self.training_end_date = "2018-91-27"

        self.incomeQ_df = {}
        self.balanceQ_df = {}
        self.valuation_df = {}

    def adftest(self):
        self.df_dict = DailyPrice().loadAll()
        for security in self.df_dict.keys():
            df = self.df_dict[security].loc[
                (self.df_dict[security].index > self.start_date)
                & (self.df_dict[security].index <= self.end_date)
            ]
            if df.empty or df["close"].count() < 8:
                continue
            X = df["close"].dropna().values
            result = adfuller(X, maxlag=1, regression="c", autolag=None)
            if result[0] < result[4]["1%"]:
                print("{0} {1} - {2}".format(security, self.start_date, self.end_date))
                print("ADF Statistic: %f" % result[0])
                print("p-value: %f" % result[1])
                print("Critical Values:")
                for key, value in result[4].items():
                    print("\t%s: %.3f" % (key, value))
                self.halflife(X)

    def cadftest(self, index="GLD"):
        self.start_date = "2020-03-10"
        self.end_date = TODAY
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
            self.security_df = Security().load(securityType="stock")
        else:
            self.df_dict = USDailyPrice().loadAll()
            self.security_df = USSecurity().load()
            
        stock_list = self.security_df["index"].to_list()
        # for security_x in self.df_dict.keys():
        # security_x = "000858.XSHE"
        for security_x in self.df_dict.keys():
            if security_x not in stock_list:
                continue
            if security_x != index:
                continue
            df_x = self.df_dict[security_x].loc[
                (self.df_dict[security_x].index > self.start_date)
                & (self.df_dict[security_x].index <= self.end_date)
            ]
            for security_y in self.df_dict.keys():
                if security_y not in stock_list:
                    continue
                if security_x == security_y:
                    continue
                if security_y != "SLV":
                    continue
                df_y = self.df_dict[security_y].loc[
                    (self.df_dict[security_y].index > self.start_date)
                    & (self.df_dict[security_y].index <= self.end_date)
                ]

                if df_y.empty:
                    continue

                df_merge = pd.concat([df_x["close"], df_y["close"]], axis=1)
                df_merge.columns = ['close_x', 'close_y']
                df_merge.iloc[0, :] = df_merge.iloc[0].fillna(0)
                df_merge.fillna(method="ffill", inplace=True)
                coint_t1, pvalue1, crit_value1 = ts.coint(df_merge["close_x"], df_merge["close_y"])
                coint_t2, pvalue2, crit_value2 = ts.coint(df_merge["close_y"], df_merge["close_x"])
                t = 0
                crit_value = []
                pair_msg = ""
                df = pd.DataFrame()
                if coint_t1 < coint_t2:
                    t = coint_t1
                    crit_value = crit_value1
                    pair_msg = "{} vs {}".format(security_x, security_y)
                    df = pd.concat([df_merge["close_x"], df_merge["close_y"]], axis=1)
                else:
                    t = coint_t2
                    crit_value = crit_value2
                    pair_msg = "{} vs {}".format(security_y, security_x)
                    df = pd.concat([df_merge["close_y"], df_merge["close_x"]], axis=1)
                if t < crit_value[0]:
                    df.fillna(method="ffill", inplace=True)
                    df.columns = ["x", "y"]
                    results = smapi.ols(formula="x ~ y", data=df).fit()
                    hedgeRatio = results.params[1]
                    print(
                        "{} {} - {} hedgeRatio={} t={} Critical Values 0:{}".format(
                            pair_msg,
                            self.start_date,
                            self.end_date,
                            hedgeRatio,
                            t,
                            crit_value[0],
                        )
                    )
                    pd.DataFrame((df["x"] - hedgeRatio * df["y"])).plot()
                    plt.show()

    def johansen(self, index = "GLD"):
        self.start_date = "2020-03-10"
        self.end_date = TODAY        
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()        

        df = self.df_dict[index]["close"]
        df = pd.concat([df, self.df_dict["SLV"]["close"]], axis=1)
        #df = pd.concat([df, self.df_dict["002182.XSHE"]["close"]], axis=1)
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]
        result = vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
        print(result.lr1)
        print(result.cvt)
        print(result.lr2)
        print(result.cvm)

        print(result.eig)  # eigenvalues
        print(result.evec)  # eigenvectors

        yport = pd.DataFrame(
            np.dot(df.values, result.evec[:, 0])
        )  #  (net) market value of portfolio
        # yport.plot()
        # plt.show()
        ylag = yport.shift()
        deltaY = yport - ylag
        df2 = pd.concat([ylag, deltaY], axis=1)
        df2.columns = ["ylag", "deltaY"]
        regress_results = smapi.ols(
            formula="deltaY ~ ylag", data=df2
        ).fit()  # Note this can deal with NaN in top row
        print(regress_results.params)

        halflife = -np.log(2) / regress_results.params["ylag"]
        print("halflife=%f days" % halflife)

        #  Apply a simple linear mean reversion strategy to EWA-EWC-IGE
        lookback = np.round(halflife).astype(
            int
        )  #  setting lookback to the halflife found above
        numUnits = (
            -(yport - yport.rolling(lookback).mean()) / yport.rolling(lookback).std()
        )  # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2
        positions = pd.DataFrame(
            np.dot(numUnits.values, np.expand_dims(result.evec[:, 0], axis=1).T)
            * df.values
        )  # results.evec(:, 1)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
        pnl = np.sum(
            (positions.shift().values) * (df.pct_change().values), axis=1
        )  # daily P&L of the strategy
        ret = pnl / np.sum(np.abs(positions.shift()), axis=1)
        pd.DataFrame((np.cumprod(1 + ret) - 1)).plot()
        print(
            "APR=%f Sharpe=%f"
            % (
                np.prod(1 + ret) ** (252 / len(ret)) - 1,
                np.sqrt(252) * np.mean(ret) / np.std(ret),
            )
        )
        plt.show()

    def index_vs_stocks(self):
        df_dict = DailyPrice().loadAll()
        df = DailyPrice().loadAll_to_df(column="close", securityType="stock")
        trainDataIdx = df.index[
            (df.index > self.training_start_date) & (df.index <= self.training_end_date)
        ]
        testDataIdx = df.index[df.index > self.start_date]
        # df.iloc[0, :] = df.iloc[0].fillna(0)
        # df=df.fillna(method='ffill')

        cl_etf = df_dict["000300.XSHG"]["close"]

        isCoint = np.full(df.shape[1], False)
        for s in range(df.shape[1]):
            # Combine the two time series into a matrix y2 for input into Johansen test
            y2 = pd.concat(
                [df.loc[trainDataIdx].iloc[:, s], cl_etf.loc[trainDataIdx]], axis=1
            )
            y2 = y2.loc[
                y2.notnull().all(axis=1),
            ]

            if y2.shape[0] > 100:
                # Johansen test
                result = vm.coint_johansen(y2.values, det_order=0, k_ar_diff=1)
                if result.lr1[0] > result.cvt[0, 2]:
                    isCoint[s] = True

        print(isCoint.sum())
        yN = df.loc[trainDataIdx, isCoint]
        logMktVal_long = np.sum(
            np.log(yN), axis=1
        )  # The net market value of the long-only portfolio is same as the "spread"

        # Confirm that the portfolio cointegrates with SPY
        ytest = pd.concat([logMktVal_long, np.log(cl_etf.loc[trainDataIdx])], axis=1)
        result = vm.coint_johansen(ytest, det_order=0, k_ar_diff=1)
        print(result.lr1)
        print(result.cvt)
        print(result.lr2)
        print(result.cvm)

        # Apply linear mean-reversion model on test set
        yNplus = pd.concat(
            [df.loc[testDataIdx, isCoint], pd.DataFrame(cl_etf.loc[testDataIdx])],
            axis=1,
        )  # Array of stock and ETF prices
        weights = np.column_stack(
            (
                np.full((testDataIdx.shape[0], isCoint.sum()), result.evec[0, 0]),
                np.full((testDataIdx.shape[0], 1), result.evec[1, 0]),
            )
        )  # Array of log market value of stocks and ETF's

        logMktVal = np.sum(
            weights * np.log(yNplus), axis=1
        )  # Log market value of long-short portfolio

        # Calculate halflife
        ylag = logMktVal.shift()
        deltaY = logMktVal - ylag
        df2 = pd.concat([ylag, deltaY], axis=1)
        df2.columns = ["ylag", "deltaY"]
        regress_results = smapi.ols(
            formula="deltaY ~ ylag", data=df2
        ).fit()  # Note this can deal with NaN in top row
        print(regress_results.params)
        halflife = -np.log(2) / regress_results.params["ylag"]
        print("halflife=%f days" % halflife)

        lookback = np.round(halflife).astype(int)
        numUnits = (
            -(logMktVal - logMktVal.rolling(lookback).mean())
            / logMktVal.rolling(lookback).std()
        )  # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2
        positions = pd.DataFrame(
            np.expand_dims(numUnits, axis=1) * weights
        )  # results.evec(:, 1)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.

        pd.DataFrame(
            (positions.shift().values)
            * (np.log(yNplus) - np.log(yNplus.shift()).values)
        ).plot()
        plt.show()

        pnl = np.sum(
            (positions.shift().values)
            * (np.log(yNplus) - np.log(yNplus.shift()).values),
            axis=1,
        )  # daily P&L of the strategy
        ret = pd.DataFrame(
            pnl.values / np.sum(np.abs(positions.shift()), axis=1).values
        )
        (np.cumprod(1 + ret) - 1).plot()
        plt.show()
        print(
            "APR=%f Sharpe=%f"
            % (
                np.prod(1 + ret) ** (252 / len(ret)) - 1,
                np.sqrt(252) * np.mean(ret) / np.std(ret),
            )
        )

    def longShortStocks(self):
        df_dict = DailyPrice().loadAll()
        df_close = DailyPrice().loadAll_to_df(column="close", securityType="stock")
        df_close.index = pd.to_datetime(df_close.index)
        df_open = DailyPrice().loadAll_to_df(column="open", securityType="stock")
        df_close.index = pd.to_datetime(df_close.index)

        df_close = df_close.loc[
            (df_close.index >= self.start_date) & (df_close.index <= self.end_date), :,
        ]
        df_close.fillna(method="ffill", inplace=True)
        df_close.fillna(0, inplace=True)
        df_close = df_close.drop([col for col in df_close.columns if df_close[col][0]==0], axis=1)
        #df_close=df_close.filter(regex ='^600')
        

        df_open = df_open.loc[
            (df_open.index >= self.start_date) & (df_open.index <= self.end_date), :,
        ]
        df_open.fillna(method="ffill", inplace=True)        
        df_open.fillna(0, inplace=True)
        df_open = df_open.drop([col for col in df_open.columns if df_open[col][0]==0], axis=1)        
        #df_open=df_open.filter(regex ='^600')
        
        ret = df_close.pct_change()  # daily returns
        marketRet = np.mean(ret, axis=1)  # equal weighted market index return

        weights = -(np.array(ret) - np.reshape(marketRet.values, (ret.shape[0], 1)))
        weights = weights / pd.DataFrame(np.abs(weights)).sum(axis=1).values.reshape(
            (weights.shape[0], 1)
        )
        weights = pd.DataFrame(
            weights, columns=df_close.columns, index=np.array(ret.index)
        )

        dailyret = (weights.shift() * ret).sum(axis=1)  # Capital is always one

        #((1 + dailyret).cumprod() - 1).plot()
        #plt.show()
        print(
            "APR=%f Sharpe=%f"
            % (
                np.prod(1 + dailyret) ** (252 / len(dailyret)) - 1,
                np.sqrt(252) * np.mean(dailyret) / np.std(dailyret),
            )
        )
        # APR=13.7%, Sharpe=1.3

        ret = (df_open - df_close.shift()) / df_close.shift()  # daily returns

        marketRet = np.mean(ret, axis=1)  # equal weighted market index return

        weights = -(np.array(ret) - np.reshape(marketRet.values, (ret.shape[0], 1)))
        weights = weights / pd.DataFrame(np.abs(weights)).sum(axis=1).values.reshape(
            (weights.shape[0], 1)
        )
        weights = pd.DataFrame(
            weights, columns=df_close.columns, index=np.array(ret.index)
        )

        #Close at current day
        #dailyret = (weights * (df_close - df_open) / df_open).sum(axis=1)  # Capital is always one
        
        #Close at next day open
        dailyret = (weights * (df_open.shift(-1) - df_open) / df_open).sum(axis=1)  # Capital is always one
        
        ((1 + dailyret).cumprod() - 1).plot()
        print(
            "APR=%f Sharpe=%f"
            % (
                np.prod(1 + dailyret) ** (252 / len(dailyret)) - 1,
                np.sqrt(252) * np.mean(dailyret) / np.std(dailyret),
            )
        )
        plt.show()

    def hurst(self):
        self.df_dict = DailyPrice().loadAll()
        for security in self.df_dict.keys():
            #            if security != "603939.XSHG":
            #                continue
            df = self.df_dict[security].loc[
                (self.df_dict[security].index > self.start_date)
                & (self.df_dict[security].index <= self.end_date)
            ]
            if df.empty or df["close"].count() < 100:
                continue
            series = df["close"].dropna()
            X = series[series > 0].tail(100).values
            if np.var(X) == 0:
                continue
            H, c, data = compute_Hc(X, kind="price", simplified=True)
            # H, pVal=genhurst(np.log(X))
            if H < 0.5:
                print(
                    "{0} {1} - {2} Hurst: {3} ".format(
                        security, self.start_date, self.end_date, H
                    )
                )
                self.halflife(X)

    def halflife(self, X):
        # half life
        X_lag = np.roll(X, 1)
        X_lag[0] = 0
        X_ret = X - X_lag
        X_ret[0] = 0
        X_lag2 = sm.add_constant(X_lag)
        model = sm.OLS(X_ret, X_lag2)
        res = model.fit()
        halflife = -math.log(2) / res.params[1]
        print("Halflife = {0}".format(halflife))
        return halflife

    def ebit(self, index):
        if not self.incomeQ_df:
            self.incomeQ_df = IncomeQuarter().loadAll()
        if index not in self.incomeQ_df:
            return 0
        ebit = (
            self.incomeQ_df[index]["net_profit"].tail(1).values / 10 ** 8
            + self.incomeQ_df[index]["interest_expense"].tail(1).values / 10 ** 8
            + self.incomeQ_df[index]["income_tax"].tail(1).values / 10 ** 8
        )
        return ebit

    def tev(self, index):
        if not self.balanceQ_df:
            self.balanceQ_df = BalanceQuarter().loadAll()
        if not self.valuation_df:
            self.valuation_df = Valuation().loadAll()
        if index not in self.balanceQ_df:
            return 0
        if index not in self.valuation_df:
            return 0
        # 市值 + 总债务 - (现金 + 流动资产 - 流动负债) + 优先股 + 少数股东权益
        tev = (
            self.valuation_df[index]["market_cap"].tail(1).values
            + self.balanceQ_df[index]["total_liability"].tail(1).values / 10 ** 8
            + self.balanceQ_df[index]["preferred_shares_equity"].tail(1).values
            / 10 ** 8
            # - self.balanceQ_df[index]["cash_equivalents"].tail(1).values/10**8
            - self.balanceQ_df[index]["total_current_assets"].tail(1).values / 10 ** 8
            + self.balanceQ_df[index]["total_current_liability"].tail(1).values
            / 10 ** 8
            + self.balanceQ_df[index]["minority_interests"].tail(1).values / 10 ** 8
        )
        return tev

    def capital(self, index):
        if index not in self.balanceQ_df:
            return 0
        capital = (
            # 净资产 = 资产总额-负债总额
            self.balanceQ_df[index]["total_assets"].tail(1).values / 10 ** 8
            - self.balanceQ_df[index]["total_liability"].tail(1).values / 10 ** 8
            # 固定资产
            + self.balanceQ_df[index]["fixed_assets"].tail(1).values / 10 ** 8
            # 净营运资本 = 流动资产 - 流动负债
            + self.balanceQ_df[index]["total_current_assets"].tail(1).values / 10 ** 8
            - self.balanceQ_df[index]["total_current_liability"].tail(1).values
            / 10 ** 8
        )
        return capital

    def percent(self):
        # 801192  银行
        # 801193  证券
        # 801050  有色
        # 801021  煤炭开采
        # 801020	采掘I
        # GN905   稀有金属
        # 801054  稀有金属
        # 801081	半导体II
        # 801101	计算机设备II
        # 801102	通信设备II
        # 801752	互联网传媒II
        self.df_dict = DailyPrice().loadAll()
        industries_df = Industry().loadAll()
        indicatorsW_df = indicators(
            class_name="WeeklyPrice", db_name="WeeklyIndicators"
        ).loadAll()
        indicatorsM_df = indicators(
            class_name="MonthlyPrice", db_name="MonthlyIndicators"
        ).loadAll()
        indicatorsD_df = indicators(
            class_name="DailyPrice", db_name="DailyIndicators"
        ).loadAll()
        moneyflow_df = MoneyFlow().loadAll()
        isSt_df = IsST().loadAll()

        for industry_index in [
            "801081",
            "801101",
            "801102",
            "801752",
        ]:
            stock_list = industries_df[industry_index][0]
            total_res_df = pd.DataFrame()
            for security in self.df_dict.keys():
                if security not in stock_list:
                    continue
                if isSt_df[security]["isSt"].tail(1).values == True:
                    continue
                df = self.df_dict[security].loc[
                    (self.df_dict[security].index > self.start_date)
                    & (self.df_dict[security].index <= self.end_date)
                ]
                min_price = df["low"].min()
                max_price = df["high"].max()
                latest_close = df["close"].iloc[-1]
                pct = (max_price - min_price) / min_price
                if pct < 999999: #0.5:
                    #if indicatorsM_df[security]["macd"].tail(1).values <= 0:
                    #    continue
                    ## if indicatorsW_df[security]["macd"].tail(1).values <= 0 :
                    ##    continue
                    #if (
                    #    indicatorsD_df[security]["sma_200"].tail(1).values
                    #    > latest_close
                    #):
                    #    continue
                    net_amount_period = (
                        moneyflow_df[security]["net_amount_main"].tail(3).sum()
                    )
                    ebit = self.ebit(security)
                    tev = self.tev(security)
                    capital = self.capital(security)
                    if ebit == 0 or tev == 0 or capital == 0:
                        continue
                    if net_amount_period > -99999999:#0:
                        res_dict = {}
                        res_dict["security"] = security
                        res_dict["ebit/tev"] = ebit / tev
                        res_dict["roc"] = ebit / capital
                        res_dict["pct"] = pct
                        res_dict["min_price"] = min_price
                        res_dict["total_net_amount"] = net_amount_period
                        res_df = pd.DataFrame.from_dict(res_dict)
                        total_res_df = total_res_df.append(res_df)
                        # print(
                        #    "{0} {1} min {2} with positive MACD M {3} W{4} net_amount {5}".format(
                        #        security,
                        #        pct,
                        #        min_price,
                        #        indicatorsM_df[security]["macd"].tail(1).values,
                        #        indicatorsW_df[security]["macd"].tail(1).values,
                        #        net_amount_period,
                        #    )
                        # )
            if total_res_df.empty:
                continue
            sorted_df = total_res_df.sort_values("roc", ascending=False)
            sorted_df.to_csv("./{0}-{1}-sorted.csv".format(TODAY,industry_index))
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):  # more options can be specified also
                print(sorted_df)

    def calEbitTevRoc(self, index):
        if index == "600519.XSHG":
            print("600519.XSHG")
        ebit = self.ebit(index)
        tev = self.tev(index)
        capital = self.capital(index)
        if ebit != 0 and tev != 0:
            self.ebittev_df = self.ebittev_df.append(
                pd.DataFrame({"index": index, "ebit/tev": ebit / tev})
            )
        if ebit != 0 and capital != 0:
            self.ebittev_df = self.ebittev_df.append(
                pd.DataFrame({"index": index, "roc": ebit / capital})
            )

    def ebittev(self):
        self.ebittev_df = pd.DataFrame()
        # self.df_dict = DailyPrice().loadAll()
        self.securities_df = Security().load()
        self.securities_df = self.securities_df.loc[
            (self.securities_df["type"] == "stock")
            & (self.securities_df["end_date"] > JQDATA_ENDDATE)
        ]

        self.securities_df["index"].apply(self.calEbitTevRoc)
        merge_df = self.securities_df.merge(
            self.ebittev_df, left_on="index", right_on="index"
        )
        # merge_df = merge_df.loc[merge_df["ebit/tev"]>0]
        merge_df = merge_df[merge_df["index"].str.contains("^600|^000")]
        # sorted_df = merge_df.sort_values("ebit/tev", ascending=False)
        sorted_df = merge_df.sort_values("roc", ascending=False)
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(sorted_df[["index", "display_name", "roc"]])

    def kalman(self, index1="SLV", index2="GLD"):
        self.start_date = "2020-01-01"
        self.end_date = TODAY          
        if isChinaMkt(index1):
            self.df_dict1 = DailyPrice().loadAll()
        else:
            self.df_dict1 = USDailyPrice().loadAll()
        if isChinaMkt(index2):
            self.df_dict2 = DailyPrice().loadAll()
        else:
            self.df_dict2 = USDailyPrice().loadAll()        
        
        df1 = self.df_dict1[index1].close
        df2 = self.df_dict2[index2].close
        df = pd.concat([df1, df2], axis=1, join="inner")
        df.columns = [index1, index2]

        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]

        x = df[index1]
        y = df[index2]

        x = np.array(ts.add_constant(x))[
            :, [1, 0]
        ]  # Augment x with ones to  accomodate possible offset in the regression between y vs x.

        delta = 0.0001  # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).

        yhat = np.full(y.shape[0], np.nan)  # measurement prediction
        e = yhat.copy()
        Q = yhat.copy()

        # For clarity, we denote R(t|t) by P(t). Initialize R, P and beta.
        R = np.zeros((2, 2))
        P = R.copy()
        beta = np.full((2, x.shape[0]), np.nan)
        Vw = delta / (1 - delta) * np.eye(2)
        Ve = 0.001

        # Initialize beta(:, 1) to zero
        beta[:, 0] = 0

        # Given initial beta and R (and P)
        for t in range(len(y)):
            if t > 0:
                beta[:, t] = beta[:, t - 1]
                R = P + Vw

            yhat[t] = np.dot(x[t, :], beta[:, t])
            #    print('FIRST: yhat[t]=', yhat[t])

            Q[t] = np.dot(np.dot(x[t, :], R), x[t, :].T) + Ve
            #    print('Q[t]=', Q[t])

            # Observe y(t)
            e[t] = y[t] - yhat[t]  # measurement prediction error
            #    print('e[t]=', e[t])
            #    print('SECOND: yhat[t]=', yhat[t])

            K = np.dot(R, x[t, :].T) / Q[t]  #  Kalman gain
            #    print(K)

            beta[:, t] = beta[:, t] + np.dot(K, e[t])  #  State update. Equation 3.11
            #    print(beta[:, t])

            P = R - np.dot(
                np.dot(K, x[t, :]), R
            )  # State covariance update. Euqation 3.12
        #    print(R)

        plt.plot(beta[0, :])
        plt.plot(beta[1, :])
        # plt.plot(e[2:])
        # plt.plot(np.sqrt(Q[2:]))
        plt.show()

        longsEntry = e < -np.sqrt(Q)
        longsExit = e > 0

        shortsEntry = e > np.sqrt(Q)
        shortsExit = e < 0

        numUnitsLong = np.zeros(longsEntry.shape)
        numUnitsLong[:] = np.nan

        numUnitsShort = np.zeros(shortsEntry.shape)
        numUnitsShort[:] = np.nan

        numUnitsLong[0] = 0
        numUnitsLong[longsEntry] = 1
        numUnitsLong[longsExit] = 0
        numUnitsLong = pd.DataFrame(numUnitsLong)
        numUnitsLong.fillna(method="ffill", inplace=True)

        numUnitsShort[0] = 0
        numUnitsShort[shortsEntry] = -1
        numUnitsShort[shortsExit] = 0
        numUnitsShort = pd.DataFrame(numUnitsShort)
        numUnitsShort.fillna(method="ffill", inplace=True)

        numUnits = numUnitsLong + numUnitsShort
        positions = pd.DataFrame(
            np.tile(numUnits.values, [1, 2])
            * ts.add_constant(-beta[0, :].T)[:, [1, 0]]
            * df.values
        )  #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
        pnl = np.sum(
            (positions.shift().values) * (df.pct_change().values), axis=1
        )  # daily P&L of the strategy
        #pnl = pd.DataFrame(pnl, index = df.index, columns=["pnl"])
        ret = pnl / np.sum(np.abs(positions.shift()), axis=1)
        (np.cumprod(1 + ret) - 1).plot()
        plt.show()
        print(
            "APR=%f Sharpe=%f"
            % (
                np.prod(1 + ret) ** (252 / len(ret)) - 1,
                np.sqrt(252) * np.mean(ret) / np.std(ret),
            )
        )
        # APR=0.313225 Sharpe=3.464060




    def autoCorrel(self, index = "GLD"):
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()
            
        self.start_date = "2020-03-19"
        self.end_date = "2020-08-07"
        df = self.df_dict[index]["close"].to_frame()
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]
        for lookback in range(1,20):
            for holddays in range(1,20):
        #for lookback in [1, 5, 10, 25, 60]:
        #    for holddays in [1, 5, 10, 25, 60]:
                ret_lag=df.pct_change(periods=lookback)
                ret_fut=df.shift(-holddays).pct_change(periods=holddays)
                if (lookback >= holddays):
                    indepSet=range(0, ret_lag.shape[0], holddays)
                else:
                    indepSet=range(0, ret_lag.shape[0], lookback)
                    
                ret_lag=ret_lag.iloc[indepSet]
                ret_fut=ret_fut.iloc[indepSet]
                goodDates=(ret_lag.notna() & ret_fut.notna()).values
                (cc, pval)=pearsonr(ret_lag[goodDates].iloc[:,0], ret_fut[goodDates].iloc[:,0])
                #if cc >= 0 and pval <= 0.1:
                print('%4i %4i %7.4f %7.4f' % (lookback, holddays, cc, pval))
        
        lookback=5
        holddays=8
        
        longs=df < df.shift(lookback)
        shorts=df > df.shift(lookback)
        
        pos=np.zeros(df.shape)
        
        for h in range(holddays-1):
            long_lag=longs.shift(h).fillna(False)
            short_lag=shorts.shift(h).fillna(False)
            pos[long_lag]=pos[long_lag]+1
            pos[short_lag]=pos[short_lag]-1
        
        pos[pos<0] = 0
        pos=pd.DataFrame(pos)
        
        pnl=np.sum(pos.shift().values * df.pct_change().values, axis=1) # daily P&L of the strategy
        ret=pnl/np.sum(np.abs(pos.shift()), axis=1)
        cumret=(np.cumprod(1+ret)-1)
        cumret.plot()
        plt.show()
        print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
        maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0))
        print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))        
        

    def momentum_top(self):
        lookback=200#252
        holddays=25#25
        topN=50
        
        self.start_date = "2015-03-19"
        self.end_date = "2020-08-01"
        #self.df_dict = DailyPrice().loadAll()
        cl = DailyPrice().loadAll_to_df(column="close", securityType="stock")
        cl.index = pd.to_datetime(cl.index)
        cl = cl[
            (cl.index > self.start_date) & (cl.index <= self.end_date)
        ]        
        ret=cl.pct_change(periods=lookback)
        longs=np.full(cl.shape, False)
        shorts=np.full(cl.shape, False)
        positions=np.zeros(cl.shape)
        
        for t in range(lookback, cl.shape[0]):
            hasData=np.where(np.isfinite(ret.iloc[t, :]))
            hasData=hasData[0]
            if len(hasData)>0:
                idxSort=np.argsort(ret.iloc[t, hasData])  
                longs[t, hasData[idxSort.values[np.arange(-np.min((topN, len(idxSort))),0)]]]=1
                shorts[t, hasData[idxSort.values[np.arange(0,topN)]]]=1
                
        longs=pd.DataFrame(longs)
        shorts=pd.DataFrame(shorts)
        
        for h in range(holddays-1):
            long_lag=longs.shift(h).fillna(False)
            short_lag=shorts.shift(h).fillna(False)
            positions[long_lag]=positions[long_lag]+1
            positions[short_lag]=positions[short_lag]-1
            
        positions=pd.DataFrame(positions)
        tmp = (positions.shift().values)*(cl.pct_change().values)
        tmp = np.nan_to_num(tmp)
        ret=pd.DataFrame(np.sum(tmp, axis=1)/(2*topN)/holddays) # daily P&L of the strategy
        ret.index = cl.index
        cumret=(np.cumprod(1+ret)-1)
        cumret.index = cl.index
        cumret.plot()
        plt.show()        
        print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
        maxDD, maxDDD, i=calculateMaxDD(cumret.fillna(0).values)
        print('Max DD=%f Max DDD in days=%i' % (maxDD, maxDDD))     
        
        
    def kelly(self, index="GLD"):
        self.start_date = "2020-01-01"
        self.end_date = "2020-08-10"
        self.df_dict = USDailyPrice().loadAll()
        df = self.df_dict[index]["close"].to_frame()
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]        
        r = df.close.pct_change()
        kelly = r.rolling(60).mean()/r.rolling(60).var()
        print (kelly)
        
        
    def volatility(self, index = "000001.XSHG"):
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()
            
        self.start_date = "2016-01-01"
        self.end_date = TODAY        
        df = self.df_dict[index]["close"].to_frame()   
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df = df[['close', 'log_rtn']].dropna(how = 'any')   
        df['moving_std_252'] = df[['log_rtn']].rolling(window=252).std()
        df['moving_std_21'] = df[['log_rtn']].rolling(window=21).std()
        fig, ax = plt.subplots(3, 1, figsize=(18, 15), 
                               sharex=True)
        
        df.close.plot(ax=ax[0])
        ax[0].set(title=index+' time series',
                  ylabel='Stock price ($)')
        
        df.log_rtn.plot(ax=ax[1])
        ax[1].set(ylabel='Log returns (%)')
        
        df.moving_std_252.plot(ax=ax[2], color='r', 
                               label='Moving Volatility 252d')
        df.moving_std_21.plot(ax=ax[2], color='g', 
                              label='Moving Volatility 21d')
        ax[2].set(ylabel='Moving Volatility',
                  xlabel='Date')
        ax[2].legend()
        
        # plt.tight_layout()
        # plt.savefig('images/ch1_im15.png')
        plt.show()        
        
    def pdf(self, index ="GLD"):
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()
            
        self.start_date = "2016-01-01"
        self.end_date = TODAY        
        df = self.df_dict[index]["close"].to_frame()   
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]    
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df = df[['close', 'log_rtn']].dropna(how = 'any')   
        
        r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
        mu = df.log_rtn.mean()
        sigma = df.log_rtn.std()
        norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)  
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # histogram
        sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])                                    
        ax[0].set_title('Distribution of MSFT returns', fontsize=16)                                                    
        ax[0].plot(r_range, norm_pdf, 'g', lw=2, 
                   label=f'N({mu:.2f}, {sigma**2:.4f})')
        ax[0].legend(loc='upper left');
        
        # Q-Q plot
        qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
        ax[1].set_title('Q-Q plot', fontsize = 16)
        
        # plt.tight_layout()
        # plt.savefig('images/ch1_im10.png')
        plt.show()        
        
        jb_test = scs.jarque_bera(df.log_rtn.values)
        
        print('---------- Descriptive Statistics ----------')
        print('Range of dates:', min(df.index.date), '-', max(df.index.date))
        print('Number of observations:', df.shape[0])
        print(f'Mean: {df.log_rtn.mean():.4f}')
        print(f'Median: {df.log_rtn.median():.4f}')
        print(f'Min: {df.log_rtn.min():.4f}')
        print(f'Max: {df.log_rtn.max():.4f}')
        print(f'Standard Deviation: {df.log_rtn.std():.4f}')
        print(f'Skewness: {df.log_rtn.skew():.4f}')
        print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}') 
        print(f'Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}')        
        #输出(统计量JB的值,P值)=(0.28220016508625245, 0.86840239542814834)，P值>指定水平0.05,接受原假设，可以认为样本数据在5%的显著水平下服从正态分布
        #输出(统计量JB的值,P值)=(1117.2762482645478, 0.0)，P值<指定水平0.05,拒绝原假设，认为样本数据在5%的显著水平下不服从正态分布
        
    def acf(self, index ="GLD"):
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()
            
        self.start_date = "2016-01-01"
        self.end_date = TODAY        
        df = self.df_dict[index]["close"].to_frame()   
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]    
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df = df[['close', 'log_rtn']].dropna(how = 'any')   
        
        N_LAGS = 50
        SIGNIFICANCE_LEVEL = 0.05          

        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        
        acf = smt.graphics.plot_acf(df.log_rtn, 
                                    lags=N_LAGS, 
                                    alpha=SIGNIFICANCE_LEVEL, ax = ax[0])        
        ax[0].set(title='Autocorrelation Plots',
                  ylabel='Log Returns')    
        
        smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
        ax[1].set(title='Autocorrelation Plots',
                  ylabel='Squared Returns')
        
        smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[2])
        ax[2].set(ylabel='Absolute Returns',
                  xlabel='Lag')
        
        # plt.tight_layout()
        # plt.savefig('images/ch1_im14.png')
        plt.show()        
        
    def correl(self, index1="GLD", index2="SLV"):
        if isChinaMkt(index1):
            self.df_dict1 = DailyPrice().loadAll()
        else:
            self.df_dict1 = USDailyPrice().loadAll()
        df1 = self.df_dict1[index1]["close"].to_frame()  
        if isChinaMkt(index2):
            self.df_dict2 = DailyPrice().loadAll()
        else:
            self.df_dict2 = USDailyPrice().loadAll()
        df2 = self.df_dict2[index2]["close"].to_frame()          

        if not isChinaMkt(index1) and isChinaMkt(index2):
            df2 = df2.shift(-1)
        
        if isChinaMkt(index1) and not isChinaMkt(index2):
            df1 = df1.shift(-1)        
        
        self.start_date = "2020-01-01"
        self.end_date = TODAY
        df = pd.DataFrame()
        df['log_rtn_x'] = np.log(df1.close/df1.close.shift(1))
        df['log_rtn_y'] = np.log(df2.close/df2.close.shift(1))
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]    
        df.dropna(how='any', axis=0, inplace=True)

        corr_coeff = df.log_rtn_x.corr(df.log_rtn_y)
        ax = sns.regplot(x='log_rtn_x', y='log_rtn_y', data=df, 
                         line_kws={'color': 'red'})
        ax.set(title=f'{index1} vs. {index2} ($\\rho$ = {corr_coeff:.2f})',
               ylabel=index2+' log returns',
               xlabel=index1+' log returns')
        
        # plt.tight_layout()
        # plt.savefig('images/ch1_im16.png')
        plt.show()        
        
        
        #corr_coeff trend with time
 
        
    def correlSearch(self, index1="GLD", index2="000001.XSHG"):
        self.start_date = "2020-01-01"
        self.end_date = TODAY        
        if isChinaMkt(index1):
            self.df_dict1 = DailyPrice().loadAll()
        else:
            self.df_dict1 = USDailyPrice().loadAll()
        df1 = self.df_dict1[index1]["close"].to_frame()  
        if isChinaMkt(index2):
            self.df_dict2 = DailyPrice().loadAll()
        else:
            self.df_dict2 = USDailyPrice().loadAll()
        
        for y in self.df_dict2.keys():
        #for y in ["002237.XSHE","601212.XSHG","600547.XSHG",
        #          "600459.XSHG","600988.XSHG","000603.XSHE",
        #          "000975.XSHE","601069.XSHG","601020.XSHG",
        #          "600489.XSHG","002155.XSHE","600766.XSHG",
        #          "601899.XSHG"]:
            
            df2 = self.df_dict2[y]["close"].to_frame()                      
            if not isChinaMkt(index1) and isChinaMkt(index2):
                df2 = df2.shift(-1)
            df = pd.DataFrame()
            df['log_rtn_x'] = np.log(df1.close/df1.close.shift(1))
            df['log_rtn_y'] = np.log(df2.close/df2.close.shift(1))
            df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]    
            df.dropna(how='any', axis=0, inplace=True)
            if df.shape[0] <= 50:
                continue
            corr_coeff = df.log_rtn_x.corr(df.log_rtn_y)
            if corr_coeff >= 0.5 or corr_coeff<=-0.5:           
                ax = sns.regplot(x='log_rtn_x', y='log_rtn_y', data=df, 
                                 line_kws={'color': 'red'})
                ax.set(title=f'{index1} vs. {y} ($\\rho$ = {corr_coeff:.2f})',
                       ylabel=y+' log returns',
                       xlabel=index1+' log returns')
                print (f'{index1} vs. {y} ($\\rho$ = {corr_coeff:.2f})')
                # plt.tight_layout()
                # plt.savefig('images/ch1_im16.png')
                #plt.show()        
        

    def priceChart(self, index = "GLD"):

        cf.go_offline()
        init_notebook_mode()           
        pio.renderers.default = "browser"
        
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()
            
        self.start_date = "2016-01-01"
        self.end_date = TODAY        
        df = self.df_dict[index]
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]    
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        
        df = df.rename(columns={"volumn": "volume"})
        
        qf = cf.QuantFig(df, title=index+"'s Stock Price", 
                         legend='top', name=index)
        qf.add_volume()
        qf.add_sma(periods=20, column='close', color='red')
        qf.add_ema(periods=20, color='green')   
        qf.iplot()


    def decompose(self, index="GLD"):
        pd.plotting.register_matplotlib_converters()
        WINDOW = 10
        self.start_date = "2016-01-01"
        self.end_date = TODAY
        
        if isChinaMkt(index):
            self.df_dict = DailyPrice().loadAll()
        else:
            self.df_dict = USDailyPrice().loadAll()        
        df = self.df_dict[index]['close'].to_frame()
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]   
        df = df.resample('W').last()
        
        df[str(WINDOW)+' rolling_mean'] = df.close.rolling(window=WINDOW).mean()
        df[str(WINDOW)+' rolling_std'] = df.close.rolling(window=WINDOW).std()
        df.plot(title=index+' Price')
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im1.png')
        plt.show()        
            
        decomposition_mul = seasonal_decompose(df.close, 
                                                   model='multiplicative'
                                                   )
        decomposition_add = seasonal_decompose(df.close, 
                                                   model='additive'
                                                   )        

        plt.rcParams.update({'figure.figsize': (10,10)})
        decomposition_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
        decomposition_add.plot().suptitle('Additive Decompose', fontsize=22)
        plt.tight_layout()
        # plt.savefig('images/ch3_im2.png')
        plt.show()     
        
        
        
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'index': 'ds', 'close': 'y'}, inplace=True)   
        train_indices = df.ds.apply(lambda x: x.year).values < 2021
        df_train = df.loc[train_indices].dropna()
        df_test = df.loc[~train_indices].reset_index(drop=True)    
        model_prophet = Prophet(seasonality_mode='multiplicative')
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_prophet.fit(df_train)        
        df_future = model_prophet.make_future_dataframe(periods=200)
        df_pred = model_prophet.predict(df_future)
        model_prophet.plot(df_pred)
        plt.tight_layout()
        #plt.savefig('images/ch3_im3.png')
        plt.show()        
        
        model_prophet.plot_components(df_pred)
        plt.tight_layout()
        #plt.savefig('images/ch3_im4.png')
        plt.show()        