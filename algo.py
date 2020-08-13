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


class algo(object):
    def __init__(self):  
        self.start_date = "2018-01-01"
        self.end_date = "2020-08-01"#datetime.now().strftime(DATE_FORMAT)
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

    def cadftest(self):
        self.df_dict = DailyPrice().loadAll()
        self.security_df = Security().load(securityType="stock")
        stock_list = self.security_df["index"].to_list()
        # for security_x in self.df_dict.keys():
        # security_x = "000858.XSHE"
        for security_x in self.df_dict.keys():
            if security_x not in stock_list:
                continue
            # if security_x != "000001.XSHG":
            #    continue
            df_x = self.df_dict[security_x].loc[
                (self.df_dict[security_x].index > self.start_date)
                & (self.df_dict[security_x].index <= self.end_date)
            ]
            # if df_x.empty or df_x["close"].count() < 8:
            #    continue

            for security_y in self.df_dict.keys():
                if security_y not in stock_list:
                    continue
                if security_x == security_y:
                    continue
                # if security_y != "600519.XSHG":
                #    continue
                df_y = self.df_dict[security_y].loc[
                    (self.df_dict[security_y].index > self.start_date)
                    & (self.df_dict[security_y].index <= self.end_date)
                ]
                df_y = df_y[df_y[["close"]] != 0]
                if df_y.empty or df_x["close"].count() != df_y["close"].count():
                    continue

                coint_t1, pvalue1, crit_value1 = ts.coint(df_x["close"], df_y["close"])
                coint_t2, pvalue2, crit_value2 = ts.coint(df_y["close"], df_x["close"])
                t = 0
                crit_value = []
                pair_msg = ""
                df = pd.DataFrame()
                if coint_t1 < coint_t2:
                    t = coint_t1
                    crit_value = crit_value1
                    pair_msg = "{} vs {}".format(security_x, security_y)
                    df = pd.concat([df_x["close"], df_y["close"]], axis=1)
                else:
                    t = coint_t2
                    crit_value = crit_value2
                    pair_msg = "{} vs {}".format(security_y, security_x)
                    df = pd.concat([df_y["close"], df_x["close"]], axis=1)
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

    def johansen(self):
        self.df_dict = DailyPrice().loadAll()
        self.security_df = Security().load(securityType="stock")
        df = self.df_dict["603877.XSHG"]["close"]
        df = pd.concat([df, self.df_dict["603677.XSHG"]["close"]], axis=1)
        df = pd.concat([df, self.df_dict["002182.XSHE"]["close"]], axis=1)
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
            "801192",
            "801193",
            "801050",
            "801021",
            "801020",
            "801054",
        ]:
            if industry_index == "801192":
                print(industry_index)
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
                if pct < 0.5:
                    if indicatorsM_df[security]["macd"].tail(1).values <= 0:
                        continue
                    # if indicatorsW_df[security]["macd"].tail(1).values <= 0 :
                    #    continue
                    if (
                        indicatorsD_df[security]["sma_200"].tail(1).values
                        > latest_close
                    ):
                        continue
                    net_amount_period = (
                        moneyflow_df[security]["net_amount_main"].tail(3).sum()
                    )
                    ebit = self.ebit(security)
                    tev = self.tev(security)
                    capital = self.capital(security)
                    if net_amount_period > 0:
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
            sorted_df.to_csv("./{0}-sorted.csv".format(TODAY))
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

    def kalman(self, index1="000858.XSHE", index2="600519.XSHG"):
        self.df_dict = DailyPrice().loadAll()
        df1 = self.df_dict[index1].close
        df2 = self.df_dict[index2].close
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




    def correl(self):
        self.start_date = "2020-03-19"
        self.end_date = "2020-08-07"
        self.df_dict = DailyPrice().loadAll()
        df = self.df_dict["000001.XSHG"]["close"].to_frame()
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
        
        lookback=6
        holddays=6
        
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
        df = self.df_dict["GLD"]["close"].to_frame()
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]        
        r = df.close.pct_change()
        kelly = r.rolling(60).mean()/r.rolling(60).var()
        print (kelly)