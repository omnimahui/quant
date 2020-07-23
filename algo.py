# encoding: UTF-8

from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
from genhurst import genhurst
from datetime import datetime, timedelta
from common import *
from price import DailyPrice
from finance import IncomeQuarter, BalanceQuarter
from indicators import Valuation, IsST, MoneyFlow, indicators
from industry import Industry
import pandas as pd


class algo(object):
    def __init__(self):
        self.start_date = "2020-01-01"
        self.end_date = datetime.now().strftime(DATE_FORMAT)
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
        # for security_x in self.df_dict.keys():
        security_x = "000001.XSHG"
        df_x = self.df_dict[security_x].loc[
            (self.df_dict[security_x].index > self.start_date)
            & (self.df_dict[security_x].index <= self.end_date)
        ]
        # if df_x.empty or df_x["close"].count() < 8:
        #    continue

        for security_y in self.df_dict.keys():
            if security_x == security_y:
                continue
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
            if coint_t1 < coint_t2:
                t = coint_t1
                crit_value = crit_value1
            else:
                t = coint_t2
                crit_value = crit_value2
            if t > crit_value[2]:
                print(
                    "{0} and {1} {2} - {3}".format(
                        security_x, security_y, self.start_date, self.end_date
                    )
                )
                print("CADF Statistic: %f" % t)
                print("Critical Values: %f" % crit_value[2])

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
        ebit = (
            self.incomeQ_df[index]["net_profit"].tail(1).values
            + self.incomeQ_df[index]["interest_expense"].tail(1).values
            + self.incomeQ_df[index]["income_tax"].tail(1).values
        )
        return ebit

    def tev(self, index):
        if not self.balanceQ_df:
            self.balanceQ_df = BalanceQuarter().loadAll()
        if not self.valuation_df:
            self.valuation_df = Valuation().loadAll()
        # 市值 + 总债务 - (现金 + 流动资产 - 流动负债) + 优先股 + 少数股东权益
        tev = (
            self.valuation_df[index]["market_cap"].tail(1).values
            + self.balanceQ_df[index]["total_liability"].tail(1).values
            + self.balanceQ_df[index]["preferred_shares_equity"].tail(1).values
            - self.balanceQ_df[index]["cash_equivalents"].tail(1).values
            - self.balanceQ_df[index]["total_current_assets"].tail(1).values
            + self.balanceQ_df[index]["total_current_liability"].tail(1).values
            + self.balanceQ_df[index]["minority_interests"].tail(1).values
        )
        return tev

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
                    if net_amount_period > 0:
                        res_dict = {}
                        res_dict["security"] = security
                        res_dict["ebit/tev"] = ebit / tev
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
            sorted_df = total_res_df.sort_values("ebit/tev", ascending=False)
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):  # more options can be specified also
                print(sorted_df)
