# encoding: UTF-8

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import sys
import json
from datetime import datetime
from time import time, sleep

from pymongo import MongoClient, ASCENDING, DESCENDING
import pandas as pd
import numpy as np

from vnpy.trader.object import BarData, TickData

# from vnpy.trader.app.ctaStrategy.ctaBase import (MINUTE_DB_NAME,
#                                                 DAILY_DB_NAME,
#                                                 TICK_DB_NAME,
#                                                 ETF_DAILY_DB_NAME)
import jqdatasdk as jq
import multiprocessing
from jqdatasdk import *
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import math
import argparse
import timeit
import dask.dataframe as ddf
import dask.array as da
from dask.multiprocessing import get
import dask
import functools
import time
import _pickle as pickle
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from genhurst import genhurst
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
import math

# 加载配置
config = open("config.json")
setting = json.load(config)

dask.config.set(scheduler="multiprocessing")


FIELDS = ["open", "high", "low", "close", "volume"]

STARTDATE = "2019-06-03"
ENDDATA = "2019-06-03"
DATE_FORMAT = "%Y-%m-%d"
JQDATA_STARTDATE = "2005-01-01"
JQDATA_ENDDATE = datetime.now().strftime(DATE_FORMAT)
DASK_NPARTITIONS = 50
MONGODB_HOST = "127.0.0.1"


def dot_to_underscore(string: str):
    return string.replace(".", "_")


def underscore_to_dot(string: str):
    return string.replace("_", ".")


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


class JQData(object):
    @staticmethod
    def connect():
        if jq.is_auth() == False:
            setting = json.load(open("config.json"))
            jq.auth(setting["jqUsername"], setting["jqPassword"])

    @staticmethod
    def disconnect():
        jq.logout()


class Security(object):
    def __init__(self):
        self.db_name = "securities"
        self.cl_name = "securities"
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]
        self.cl = self.db[self.cl_name]
        self.securities_df = self.load()
        # self.securities_df["index" == index]

    def update(self):
        self.cl.drop()
        securities_df = jq.get_all_securities(types=["stock", "fund", "index", "etf"])
        securities_df.reset_index(level=0, inplace=True)
        self.cl.insert_many(securities_df.to_dict("records"))
        return 1

    def load(self, securityType: str = ""):
        find_string = {"type": securityType} if securityType != "" else {}
        return pd.DataFrame(list(self.cl.find(find_string)))

    def getSecurityDate(self, index):
        row = self.securities_df[self.securities_df["index"] == index]
        if row.start_date.values[0] > np.datetime64(JQDATA_STARTDATE):
            start_date = np.datetime_as_string(row.start_date, "D").item(0)
        else:
            start_date = JQDATA_STARTDATE

        if row.end_date.values[0] < np.datetime64(JQDATA_ENDDATE):
            end_date = np.datetime_as_string(row.end_date, "D").item(0)
        else:
            end_date = JQDATA_ENDDATE
        start_date = datetime.strptime(start_date, DATE_FORMAT)
        end_date = datetime.strptime(end_date, DATE_FORMAT)
        return start_date, end_date


class SecurityBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, db_name):
        self.security = Security()
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[db_name]
        self.df_dict = {}
        self.securityType = ""
        self.index_column = ""
        self.fieldsFromDb = ["index"]

    def loadFieldsFromDB(self, fields=["index"]):
        self.df_dict = dict.fromkeys(self.db.list_collection_names(), pd.DataFrame())
        filter = {}
        filter["_id"] = 0
        for field in fields:
            filter[field] = 1
        for security in self.df_dict.keys():
            self.df_dict[security] = pd.DataFrame(
                list(self.db[security].find({}, filter).sort("index", 1))
            )

    @abstractmethod
    def query(self, index, start_date, end_date):
        pass

    def set_index_column(self, df, column: str = ""):
        df["index"] = df.index if column == "" else df[column]

    def updateOne(self, index: str):
        start_date, end_date = self.security.getSecurityDate(index)

        # Get last date of the stock from db['close']
        try:
            last_date = self.df_dict[index]["index"].iloc[-1]
            # Timestamp to datetime
            start_date = (last_date + timedelta(days=1)).to_pydatetime()
        except Exception:
            ...

        if start_date <= end_date:
            df = self.query(index, start_date, end_date)
            if not df.empty:
                self.set_index_column(df, self.index_column)
                self.db[index].insert_many(df.to_dict("records"))
                print(
                    "{0} {1} to {2} {3} downloaded".format(
                        index,
                        start_date.strftime(DATE_FORMAT),
                        end_date.strftime(DATE_FORMAT),
                        self.__class__.__name__,
                    )
                )

    def updateAll(self):
        security_df = self.security.load(securityType=self.securityType)
        self.loadFieldsFromDB(fields=self.fieldsFromDb)
        security_df["index"].apply(self.updateOne)


class DailyPrice(SecurityBase):
    def __init__(self,db_name="DailyPrice"):
        super(DailyPrice, self).__init__(db_name)
        self.price_pickle = "dailyprice.pkl"

    def query(self, index, start_date, end_date):
        df = jq.get_price(
            index, start_date, end_date, skip_paused=True, fill_paused=False
        )
        return df

    def loadAll(self):
        try:
            fp = open(self.price_pickle, "rb")
            self.df_dict = pickle.load(fp)
        except Exception:
            self.df_dict = dict.fromkeys(
                self.db.list_collection_names(), pd.DataFrame()
            )
            for security in self.df_dict.keys():
                self.df_dict[security] = pd.DataFrame(
                    list(self.db[security].find({}, {"_id": 0}).sort("index", 1))
                )
                self.df_dict[security].index = self.df_dict[security]["index"]
            fp = open(self.price_pickle, "wb")
            pickle.dump(self.df_dict, fp)

        return self.df_dict


class WeeklyPrice(DailyPrice):
    def __init__(self, db_name="WeeklyPrice"):
        super(WeeklyPrice, self).__init__(db_name)
        self.price_pickle = "weeklyprice.pkl"
        self.dailyprice_df = DailyPrice().loadAll()

    def query(self, index, start_date, end_date):
        if self.dailyprice_df.get(index) is None:
            return pd.DataFrame()
        stock_weekly_df = (
            self.dailyprice_df[index][start_date:end_date]
            .resample("W")
            .agg(
                {
                    "close": take_last,
                    "high": "max",
                    "low": "min",
                    "money": "sum",
                    "open": take_first,
                    "volume": "sum",
                }
            )
        )
        stock_weekly_df.index = stock_weekly_df.index + pd.DateOffset(days=-2)
        stock_weekly_df = stock_weekly_df.dropna()
        stock_weekly_df["index"] = stock_weekly_df.index
        return stock_weekly_df

    def updateOne(self, index: str):
        start_date, end_date = self.security.getSecurityDate(index)

        if start_date <= end_date:
            # Load all daily price to calculate weekly price
            df = self.query(index, start_date, end_date)
            if not df.empty:
                # Due to the weekly price on weekday could be incorrect,
                # We will update all weekly price anyway                
                self.db[index].drop()
                self.db[index].insert_many(df.to_dict("records"))
                print(
                    "{0} {1} to {2} {3} downloaded".format(
                        index,
                        start_date.strftime(DATE_FORMAT),
                        end_date.strftime(DATE_FORMAT),
                        self.__class__.__name__,
                    )
                )


class MonthlyPrice(WeeklyPrice):
    def __init__(self, db_name="MonthlyPrice"):
        super(MonthlyPrice, self).__init__(db_name)
        self.price_pickle = "monthlyprice.pkl"
        self.dailyprice_df = DailyPrice().loadAll()

    def query(self, index, start_date, end_date):
        if self.dailyprice_df.get(index) is None:
            return pd.DataFrame()
        stock_weekly_df = (
            self.dailyprice_df[index][start_date:end_date]
            .resample("M")
            .agg(
                {
                    "close": take_last,
                    "high": "max",
                    "low": "min",
                    "money": "sum",
                    "open": take_first,
                    "volume": "sum",
                }
            )
        )
        stock_weekly_df.index = stock_weekly_df.index + pd.DateOffset(days=-1)
        stock_weekly_df = stock_weekly_df.dropna()
        stock_weekly_df["index"] = stock_weekly_df.index
        return stock_weekly_df


class IsST(SecurityBase):
    def __init__(self):
        super(IsST, self).__init__("IsST")
        self.securityType = "stock"

    def set_index_column(self, df, column: str = ""):
        df.columns = ["isSt"]
        df["index"] = df.index if column == "" else df[column]

    def query(self, index, start_date, end_date):
        df = jq.get_extras("is_st", [index], start_date, end_date, df=True)
        return df


class MTSS(SecurityBase):
    def __init__(self):
        super(MTSS, self).__init__("MTSS")
        self.securityType = "stock"
        self.index_column = "date"

    def query(self, index, start_date, end_date):
        df = get_mtss(index, start_date, end_date)
        return df


class MoneyFlow(SecurityBase):
    def __init__(self):
        super(MoneyFlow, self).__init__("MoneyFlow")
        self.securityType = "stock"
        self.index_column = "date"

    def query(self, index, start_date, end_date):
        df = get_money_flow(index, start_date, end_date)
        return df


class FundamentalQuarter(SecurityBase):
    def __init__(self, db_name="FundamentalQuarter"):
        super(FundamentalQuarter, self).__init__(db_name)
        self.securityType = "stock"
        self.index_column = "day"
        self.fieldsFromDb = ["index", "statDate"]
        self.start_from = "2004-09-30"

    def prev_statdate(self, statdate):
        if statdate.month < 4:
            return datetime(statdate.year - 1, 12, 31)
        elif statdate.month < 7:
            return datetime(statdate.year, 3, 31)
        elif statdate.month < 10:
            return datetime(statdate.year, 6, 30)
        return datetime(statdate.year, 9, 30)

    def query(self, index, start_date, end_date):
        # Get the latest fundamental on end_date
        q = query(indicator).filter(indicator.code == index,)
        df = get_fundamentals(q, date=end_date)
        if not df.empty:
            df = df.drop(columns=["id"])
            # Change str to datetime
            df.index = df["day"]
        return df

    def updateOne(self, index):
        start_date, end_date = self.security.getSecurityDate(index)
        last_statdate = datetime.strptime(self.start_from, DATE_FORMAT)

        # Get last date of the stock from db['close']
        try:
            last_date = self.df_dict[index]["index"].iloc[-1]
            last_statdate = (
                self.df_dict[index]
                .loc[self.df_dict[index]["index"] == last_date]["statDate"]
                .values[0]
            )
            last_statdate = datetime.strptime(last_statdate, DATE_FORMAT)
            # Timestamp to datetime
            start_date = (last_date + timedelta(days=1)).to_pydatetime()
        except Exception:
            ...

        orig_start_date = start_date
        if start_date <= end_date:
            # Get end_date (today) fundamental
            df = self.query(index, start_date, end_date)
            if not df.empty:
                latest_pubdate = datetime.strptime(df["pubDate"][0], DATE_FORMAT)
                statdate = datetime.strptime(df["statDate"][0], DATE_FORMAT)
                if statdate > last_statdate:
                    start_date = latest_pubdate

                accumulative_df = pd.DataFrame()
                while statdate >= last_statdate:
                    delta = end_date - start_date
                    delta_days = delta.days + 1
                    newdf = pd.DataFrame(np.repeat(df.values, delta_days, axis=0))
                    newdf.columns = df.columns
                    ts = pd.date_range(end_date, periods=delta_days, freq="-1D")
                    newdf["day"] = ts
                    self.set_index_column(newdf, self.index_column)
                    accumulative_df = accumulative_df.append(newdf)
                    if statdate == last_statdate:
                        self.db[index].insert_many(accumulative_df.to_dict("records"))
                        break

                    statdate = self.prev_statdate(statdate)
                    end_date = start_date - timedelta(days=1)
                    df = self.query(index, start_date, end_date)
                    if not df.empty:
                        latest_pubdate = datetime.strptime(
                            df["pubDate"][0], DATE_FORMAT
                        )
                        statdate = datetime.strptime(df["statDate"][0], DATE_FORMAT)
                        if statdate > last_statdate:
                            start_date = latest_pubdate
                        else:
                            start_date = orig_start_date


class FundamentalYear(FundamentalQuarter):
    def __init__(self):
        super(FundamentalYear, self).__init__(db_name="FundamentalYear")
        self.securityType = "stock"
        self.index_column = "day"
        self.fieldsFromDb = ["index", "statDate"]
        self.start_from = "2004-12-31"

    def prev_year_statdate(self, statdate):
        return datetime(statdate.year - 1, 12, 31)

    def query(self, index, start_date, end_date):
        # Get the latest fundamental on end_date
        q = query(indicator).filter(indicator.code == index,)
        while 1:
            df = get_fundamentals(q, date=end_date)
            if df.empty:
                break
            df = df.drop(columns=["id"])
            # Change str to datetime
            df.index = df["day"]
            pubDate = datetime.strptime(df["pubDate"][0], DATE_FORMAT)
            statDate = datetime.strptime(df["statDate"][0], DATE_FORMAT)
            if statDate.month != 12:
                # Prev quarter report
                end_date = pubDate - timedelta(days=1)
            else:
                break
        return df

    def updateOne(self, index):
        start_date, end_date = self.security.getSecurityDate(index)
        last_statdate = datetime.strptime(self.start_from, DATE_FORMAT)

        # Get last date of the stock from db['close']
        try:
            last_date = self.df_dict[index]["index"].iloc[-1]
            last_statdate = (
                self.df_dict[index]
                .loc[self.df_dict[index]["index"] == last_date]["statDate"]
                .values[0]
            )
            last_statdate = datetime.strptime(last_statdate, DATE_FORMAT)
            # Timestamp to datetime
            start_date = (last_date + timedelta(days=1)).to_pydatetime()
        except Exception:
            ...

        orig_start_date = start_date
        if start_date <= end_date:
            # Get end_date (today) fundamental
            df = self.query(index, start_date, end_date)
            if not df.empty:
                latest_pubdate = datetime.strptime(df["pubDate"][0], DATE_FORMAT)
                statdate = datetime.strptime(df["statDate"][0], DATE_FORMAT)
                if statdate > last_statdate:
                    start_date = latest_pubdate

                accumulative_df = pd.DataFrame()
                while statdate >= last_statdate:
                    delta = end_date - start_date
                    delta_days = delta.days + 1
                    newdf = pd.DataFrame(np.repeat(df.values, delta_days, axis=0))
                    newdf.columns = df.columns
                    ts = pd.date_range(end_date, periods=delta_days, freq="-1D")
                    newdf["day"] = ts
                    self.set_index_column(newdf, self.index_column)
                    accumulative_df = accumulative_df.append(newdf)
                    if statdate == last_statdate:
                        self.db[index].insert_many(accumulative_df.to_dict("records"))
                        break

                    statdate = self.prev_statdate(statdate)
                    end_date = start_date - timedelta(days=1)
                    df = self.query(index, start_date, end_date)
                    if not df.empty:
                        latest_pubdate = datetime.strptime(
                            df["pubDate"][0], DATE_FORMAT
                        )
                        statdate = datetime.strptime(df["statDate"][0], DATE_FORMAT)
                        if statdate > last_statdate:
                            start_date = latest_pubdate
                        else:
                            start_date = orig_start_date
                    else:
                        self.db[index].insert_many(accumulative_df.to_dict("records"))
                        break


class algo(object):
    def __init__(self):
        self.start_date = "2020-03-23"
        self.end_date = "2020-07-08"
        self.df_dict = DailyPrice().loadAll()

    def adftest(self):
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


class indicators(object):
    def __init__(self, class_name = "DailyPrice",db_name = "DailyIndicators"):
        #Load price timeseries
        self.df_dict = eval(class_name)().loadAll()
        self.db_name = db_name
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]        

    def sma(self, index, window):
        ts = self.df_dict[index].close
        sma = ts.rolling(window=window, center=False).mean()
        sma.name = "sma_" + str(window)
        #sma = sma.reindex(ts.index)
        return sma
    
    def tr(self, index):
        df = pd.DataFrame()
        df["1"] = (self.df_dict[index].high - self.df_dict[index].low).abs()
        df["2"] = (self.df_dict[index].high - self.df_dict[index].close.shift(1)).abs()
        df["3"] = (self.df_dict[index].low - self.df_dict[index].close.shift(1)).abs()
        tr = df.max(axis=1)    
        return tr

    def atr(self, index, window):
        tr = self.tr(index)
        atr = tr.rolling(window=window, center=False).mean()
        atr.name = "atr_" + str(window)
        #atr=atr.reindex(self.df_dict[index].index)
        return atr

    def ewma(self, ts, window=9):
        ewmaSeries = ts.ewm(span=window, adjust=False).mean()
        return ewmaSeries
    
    def dif(self, index, fast=12, slow=26):
        ts = self.df_dict[index].close
        dif = self.ewma(ts, window=fast) - self.ewma(
            ts, window=slow
        )
        dif.name = "dif"
        return dif

    def dea(self, dif, window=9):
        dea = ewmaCal(dif, window=window)
        dea.name = "dea"
        return dea
    
    def macd(self, dif, dea):
        macd = (dif - dea) * 2
        macd.name = "macd"
        return macd
        
    def update(self):
        #For each index        
        #Calculate all indicaters
        for security in self.df_dict.keys():
            try:               
                sma_21 = self.sma(security, 21)
                sma_55 = self.sma(security, 55)
                sma_200 = self.sma(security, 200)
                atr_20 = self.atr(security, 20)
                atr_60 = self.atr(security, 60)
                dif = self.dif(security)
                dea = self.dea(dif)
                macd = self.macd(dif, dea)
                df = pd.DataFrame(index=self.df_dict[security].index)
                for series in [
                    sma_21,
                    sma_55,
                    sma_200,
                    atr_20,
                    atr_60,
                    dif,
                    dea,
                    macd,
                ]:            
                    df = df.merge(series.to_frame(), left_index=True, right_index=True)
    
                df["index"] = self.df_dict[security].index
                #Insert into DB
                self.db[security].drop()
                self.db[security].insert_many(df.to_dict("records"))

            except Exception as ex:
                print ("Got exception: {0} {1}".format(security, ex))

def test_print(value):
    print(value)


def get_industry_stocks(code, date):
    stocks = jq.get_industry_stocks(code, date)
    return stocks


INDUSTRIES = ["zjw", "sw_l1", "sw_l2", "sw_l3", "jq_l1", "jq_l2"]


def get_all_industries_on_date(date):
    for industry_category in INDUSTRIES:
        all_industries_db = mc["all_industries_" + industry_category]
        jq_industries = jq.get_industries(industry_category, date)
        if jq_industries.empty:
            continue
        jq_industries["code"] = jq_industries.index
        jq_industries["stocks"] = jq_industries["code"].apply(
            get_industry_stocks, date=date
        )
        all_industries_db[date.strftime("%Y")].insert_many(
            jq_industries.drop(columns=["start_date"]).to_dict("records")
        )


def get_industries():
    all_trade_days = jq.get_all_trade_days()
    years_processed = list()
    for date in all_trade_days:
        if date.year not in years_processed:
            print("{} processing".format(date.year))
            get_all_industries_on_date(date)
            years_processed.append(date.year)


def get_concept_stocks(code, date):
    stocks = jq.get_concept_stocks(code, date)
    return stocks


def get_all_concepts_on_date(date, jq_concepts_list):
    all_concepts_db = mc["all_concepts"]
    jq_concepts = jq_concepts_list[0]
    jq_concepts["code"] = jq_concepts.index
    jq_concepts["stocks"] = jq_concepts["code"].apply(get_concept_stocks, date=date)
    all_concepts_db[date.strftime("%Y")].insert_many(
        jq_concepts.drop(columns=["start_date"]).to_dict("records")
    )


def get_concepts():
    jq_concepts = jq.get_concepts()
    all_trade_days = jq.get_all_trade_days()
    years_processed = list()
    for date in all_trade_days:
        if date.year not in years_processed:
            print("{} processing".format(date.year))
            get_all_concepts_on_date(date, [jq_concepts])
            years_processed.append(date.year)


def smaCal(close_price_series, window):
    sma = close_price_series.rolling(window=window, center=False).mean()
    sma.name = "sma_" + str(window)
    sma = sma.reindex(close_price_series.index)
    return sma


def trCal(stock):
    # tr = pd.Series(0.0,index=stock.index)
    df = pd.DataFrame()
    df["1"] = (stock.high - stock.low).abs()
    df["2"] = (stock.high - stock.close.shift(1)).abs()
    df["3"] = (stock.low - stock.close.shift(1)).abs()
    tr = df.max(axis=1)
    # tr=tr.reindex(stock.index)
    return tr


def atrCal(stock, window):
    tr = trCal(stock)
    atr = tr.rolling(window=window, center=False).mean()
    atr.name = "atr_" + str(window)
    # atr=atr.reindex(stock.index)
    return atr


def momentumCalInWindow(close_price_series):
    logArray = np.log(close_price_series.values)
    x = pd.Series(np.arange(logArray.size)).to_numpy().reshape(-1, 1)
    logArray = logArray.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x, logArray)  # perform linear regression
    slope = linear_regressor.coef_
    r2 = linear_regressor.score(x, logArray)
    annual_slope = math.pow(np.exp(slope), 250) - 1
    adj_slope = annual_slope * r2
    return adj_slope


# @timer
def momentumCal(close_price_series, window):
    adj_slope = close_price_series.rolling(window=window, center=False).apply(
        momentumCalInWindow, raw=False
    )
    adj_slope.name = "momentum_" + str(window)
    adj_slope = adj_slope.reindex(close_price_series.index)
    return adj_slope


def momentumCalInWindow_old(logArray):
    x = pd.Series(np.arange(logArray.size)).to_numpy().reshape(-1, 1)
    logArray = logArray.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x, logArray)  # perform linear regression
    slope = linear_regressor.coef_
    r2 = linear_regressor.score(x, logArray)
    annual_slope = math.pow(np.exp(slope), 250) - 1
    adj_slope = annual_slope * r2
    return adj_slope


# @timer
def momentumCal_old(stock, window):
    log = pd.Series(list(np.log(stock.close)), index=stock["index"])
    adj_slope = pd.Series(0.0, index=stock["index"])
    adj_slope = log.rolling(window=window, center=False).apply(
        momentumCalInWindow_old, raw=True
    )
    adj_slope.name = "momentum_" + str(window)
    return adj_slope


def jqstock_to_stock(stock):
    index = stock.find(".")
    return stock[:index] if index > -1 else stock


def stock_to_jqstock(stock):
    index = stock.find(".")
    if index == -1:
        return stock + ".XSHG" if int(stock) >= 600000 else stock + ".XSHE"
    else:
        return stock


def update_indicator_old(date, stock, series):
    stock_indicator = mc["stock_indicator"]
    stock_indicator[series.name].update_one(
        {"index": date}, {"$set": {jqstock_to_stock(stock): series[date]}}, upsert=True
    )


def update_indicator(date, stock, series):
    stock_indicator = mc["stock_indicator"]
    stock_indicator.indicators.update_one(
        {"$and": [{"index": date}, {"code": jqstock_to_stock(stock)}]},
        {series.to_dict("records")},
        upsert=True,
    )


def update_indicators(series, stock):
    #    date_series = pd.Series(series.index,index=series.index)
    #    date_series.apply(update_indicator, stock=stock, series=series)
    stock_indicator = mc["stock_indicator"]
    #    df = pd.DataFrame(list(stock_indicator.indicators.find({"$and": [{'date': series['date'].to_pydatetime()},{'code':series['code']}]})))
    #    stock_indicator.indicators.update_one({"$and": [{'date': series['date'].to_pydatetime()},{'code':series['code']}]},
    #                                            {"$set": series.to_dict()},
    #                                            upsert = True)
    #    if df.empty:
    stock_indicator.indicators.insert_one(series.to_dict())


def update_index_indicator(date, stock, series):
    index_indicator = mc["index_indicator"]
    index_indicator.indicators.update_one(
        {"$and": [{"index": date}, {"code": jqstock_to_stock(stock)}]},
        {series.to_dict("records")},
        upsert=True,
    )


def update_index_indicators(series, stock):
    #    date_series = pd.Series(series.index,index=series.index)
    #    date_series.apply(update_indicator, stock=stock, series=series)
    index_indicator = mc["index_indicator"]
    #    df = pd.DataFrame(list(stock_indicator.indicators.find({"$and": [{'date': series['date'].to_pydatetime()},{'code':series['code']}]})))
    #    stock_indicator.indicators.update_one({"$and": [{'date': series['date'].to_pydatetime()},{'code':series['code']}]},
    #                                            {"$set": series.to_dict()},
    #                                            upsert = True)
    #    if df.empty:
    index_indicator.indicators.insert_one(series.to_dict())


def difCal(close_price_series, fast=12, slow=26):
    dif = ewmaCal(close_price_series, window=fast) - ewmaCal(
        close_price_series, window=slow
    )
    dif.name = "dif"
    return dif


def deaCal(dif, window=9):
    dea = ewmaCal(dif, window=window)
    dea.name = "dea"
    return dea


def ewmaCal(stockSeries, window=9):
    ewmaSeries = stockSeries.ewm(span=window, adjust=False).mean()
    return ewmaSeries


def new_round(_float, _len=2):
    if str(_float)[-1] == "5":
        return round(float(str(_float)[:-1] + "6"), _len)
    else:
        return round(_float, _len)


def uplimitCalInWindow(closePriceInWindow, prev_closePriceInWindow):
    uplimitPriceInWindow = prev_closePriceInWindow * 1.1
    uplimitPriceInWindow = np.array(list(map(new_round, uplimitPriceInWindow)))
    num = np.count_nonzero(np.equal(uplimitPriceInWindow, closePriceInWindow))
    if num > 0:
        s = 1
    return num


def uplimitCal(stock_df, window):
    uplimit = stock_df.close.rolling(window=window, center=False).apply(
        lambda x: uplimitCalInWindow(
            stock_df.loc[x.index, "close"], stock_df.loc[x.index, "pre_close"]
        ),
        raw=False,
    )
    uplimit.name = "uplimitNumPast_" + str(window)
    uplimit = uplimit.reindex(stock_df.index)
    return uplimit


def stockCal_worker(in_queue, worker_index):
    print("worker {}".format(worker_index))
    stock_price_db = mc["stock_price"]
    while True:
        stock = in_queue.get()
        if stock is None:
            break
        stock_df = pd.DataFrame(
            list(stock_price_db[stock].find().sort("index", ASCENDING))
        )
        stock_df.set_index(["index"], inplace=True)
        momentum_60 = momentumCal(stock_df.close, 60)

        #        momentum_60_old = momentumCal_old(stock_df, 60)
        # Can not use dask due to window rolling (not row by row)
        #       momentum_60_dask = ddf.from_pandas(close_price_df,npartitions=16).\
        #            apply(momentumCal, window=60).\
        #            compute()
        momentum_20 = momentumCal(stock_df.close, 20)
        momentum_10 = momentumCal(stock_df.close, 10)
        momentum_5 = momentumCal(stock_df.close, 5)
        sma_21 = smaCal(stock_df.close, 21)
        sma_55 = smaCal(stock_df.close, 55)
        sma_200 = smaCal(stock_df.close, 200)
        atr_20 = atrCal(stock_df, 20)
        atr_60 = atrCal(stock_df, 60)
        dif = difCal(stock_df.close)
        dea = deaCal(dif)
        macd = (dif - dea) * 2
        macd.name = "macd"
        stock_df["pre_close"] = stock_df.close.shift(1)
        # reversed_stock_df = stock_df.sort_index(ascending=False)
        uplimit_past_5 = uplimitCal(stock_df, 5)
        uplimit_past_10 = uplimitCal(stock_df, 10)
        uplimit_past_20 = uplimitCal(stock_df, 20)
        df = pd.DataFrame(index=stock_df.index)
        for series in [
            momentum_60,
            momentum_20,
            momentum_10,
            momentum_5,
            sma_21,
            sma_55,
            sma_200,
            atr_20,
            atr_60,
            dif,
            dea,
            macd,
            uplimit_past_5,
            uplimit_past_10,
            uplimit_past_20,
        ]:
            df = df.merge(series.to_frame(), left_index=True, right_index=True)
        df["date"] = df.index
        df["code"] = jqstock_to_stock(stock)
        df.apply(update_indicators, stock=stock, axis=1)
        print("worker {} {} done".format(worker_index, stock))

    in_queue.task_done()
    return 0


def indexCal_worker(in_queue, worker_index):
    print("worker {}".format(worker_index))
    index_price_db = mc["index_price"]
    while True:
        index = in_queue.get()
        if index is None:
            break
        index_df = pd.DataFrame(
            list(index_price_db[index].find().sort("index", ASCENDING))
        )
        index_df.set_index(["index"], inplace=True)
        momentum_60 = momentumCal(index_df.close, 60)
        momentum_20 = momentumCal(index_df.close, 20)
        sma_21 = smaCal(index_df.close, 21)
        sma_55 = smaCal(index_df.close, 55)
        sma_200 = smaCal(index_df.close, 200)
        atr_20 = atrCal(index_df, 20)
        atr_60 = atrCal(index_df, 60)
        dif = difCal(index_df.close)
        dea = deaCal(dif)
        macd = (dif - dea) * 2
        macd.name = "macd"
        df = pd.DataFrame(index=index_df.index)
        for series in [
            momentum_60,
            momentum_20,
            sma_21,
            sma_55,
            sma_200,
            atr_20,
            atr_60,
            dif,
            dea,
            macd,
        ]:
            df = df.merge(series.to_frame(), left_index=True, right_index=True)
        df["date"] = df.index
        df["code"] = index
        df.apply(update_index_indicators, stock=index, axis=1)
        print("worker {} {} done".format(worker_index, index))

    in_queue.task_done()
    return 0


def timing(method):
    def timed(*args, **kw):
        ts = timeit.default_timer()
        result = method(*args, **kw)
        te = timeit.default_timer()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timing
def stockCal():
    num_worker_threads = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    worker_queue = manager.Queue()
    stock_price_db = mc["stock_price"]
    stocks = stock_price_db.list_collection_names()
    stocks.sort()
    stock_indicator_db = mc["stock_indicator"]
    stock_indicator_db.indicators.drop()
    for stock in stocks:
        worker_queue.put(stock)

    pool = multiprocessing.Pool(num_worker_threads)
    for i in range(num_worker_threads):
        pool.apply_async(stockCal_worker, args=(worker_queue, i))

    # stop workers
    for i in range(num_worker_threads):
        worker_queue.put(None)

    pool.close()
    pool.join()


@timing
def indexCal():
    num_worker_threads = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    worker_queue = manager.Queue()
    index_price_db = mc["index_price"]
    indexes = index_price_db.list_collection_names()
    indexes.sort()
    index_indicator_db = mc["index_indicator"]
    index_indicator_db.indicators.drop()
    for index in indexes:
        worker_queue.put(index)

    pool = multiprocessing.Pool(num_worker_threads)
    for i in range(num_worker_threads):
        pool.apply_async(indexCal_worker, args=(worker_queue, i))

    # stop workers
    for i in range(num_worker_threads):
        worker_queue.put(None)

    pool.close()
    pool.join()


def get_industry_price(industry_code, year):
    industry_price_db = mc["industry_price"]
    q = query(finance.SW1_DAILY_PRICE).filter(
        finance.SW1_DAILY_PRICE.code == industry_code,
        finance.SW1_DAILY_PRICE.date >= year + "-01-01",
        finance.SW1_DAILY_PRICE.date <= year + "-12-31",
    )
    price = finance.run_query(q)

    if not price.empty:
        price["index"] = price.index
        price["date"] = price["date"].astype(str)
        industry_price_db[industry_code].insert_many(price.to_dict("records"))
        print("{0} {1} downloaded".format(industry_code, year))


def uptodate_industry_price():
    industry_price_db = mc["industry_price"]
    # Get last date of the stock from db
    industries_code = industry_price_db.list_collection_names()
    industries_code.sort()
    for industry_code in industries_code:
        last_date_of_db = list(
            industry_price_db[industry_code]
            .find({}, {"date": 1, "_id": 0})
            .sort("date", -1)
            .limit(1)
        )[0]["date"]
        q = query(finance.SW1_DAILY_PRICE).filter(
            finance.SW1_DAILY_PRICE.code == industry_code,
            finance.SW1_DAILY_PRICE.date > last_date_of_db,
        )
        price = finance.run_query(q)
        if not price.empty:
            price["index"] = price.index
            price["date"] = price["date"].astype(str)
            industry_price_db[industry_code].insert_many(price.to_dict("records"))
            print("{0} after {1} downloaded".format(industry_code, last_date_of_db))


def get_industries_price():
    all_industries_sw_l1 = mc["all_industries_sw_l1"]
    years = all_industries_sw_l1.list_collection_names()
    for y in years:
        all_industries = pd.DataFrame(list(all_industries_sw_l1[y].find()))
        all_industries.code.apply(get_industry_price, year=y)


def convertStockPrice():
    stock_price_db = mc["stock_price"]
    stock_price_db.stock_price.drop()
    stocks = sorted(stock_price_db.list_collection_names())
    for stock in stocks:
        df = pd.DataFrame(list(stock_price_db[stock].find({})))
        df["code"] = stock
        stock_price_db.stock_price.insert_many(df.to_dict("records"))
        print("{} done".format(stock))


def convertIndexPrice():
    index_price_db = mc["index_price"]
    index_price_db.index_price.drop()
    stocks = sorted(index_price_db.list_collection_names())
    for stock in stocks:
        df = pd.DataFrame(list(index_price_db[stock].find({})))
        df["code"] = stock
        index_price_db.index_price.insert_many(df.to_dict("records"))
        print("{} done".format(stock))


def loadprice():
    stock_price_db = mc["stock_price"]
    df = pd.DataFrame(list(stock_price_db.stock_price.find({})))
    print("loaded")


def stock_value_on_date(date, stock):
    stock_valuation_db = mc["stock_valuation"]
    q = query(valuation).filter(valuation.code == stock)
    valuation_df = get_fundamentals(q, date.strftime(DATE_FORMAT))
    if not valuation_df.empty:
        result = stock_valuation_db[date.strftime(DATE_FORMAT)].replace_one(
            {"code": stock}, valuation_df.to_dict("records")[0], upsert=True
        )


# def update_stock_valuation(date,stock,df):
#    stock_valuation_db = mc["stock_valuation"]
#    df=df.rename(columns = {'code.1':'code','day.1':'day'})
#    result=stock_valuation_db[date].replace_one(
#        {'code':stock},
#        df[df['day'] == date].to_dict("records")[0],
#        upsert=True)


def stock_value_by_date(stock):
    stock_price_db = mc["stock_price"]
    stock_valuation_db = mc["stock_valuation"]

    first_date_of_stock = list(
        stock_price_db[stock].find({}, {"index": 1, "_id": 0}).sort("index", 1).limit(1)
    )[0]["index"]
    last_date_of_stock = list(
        stock_price_db[stock]
        .find({}, {"index": 1, "_id": 0})
        .sort("index", -1)
        .limit(1)
    )[0]["index"]
    result = list(
        stock_valuation_db.valuation.find({"code": stock}, {"date": 1, "_id": 0})
        .sort("date", -1)
        .limit(1)
    )
    if result:
        last_date_of_stock_valuation_db = datetime.strptime(
            result[0]["date"], DATE_FORMAT
        )
        if last_date_of_stock_valuation_db == last_date_of_stock:
            return
        if last_date_of_stock_valuation_db > first_date_of_stock:
            first_date_of_stock = last_date_of_stock_valuation_db
    q = query(valuation).filter(valuation.code == stock)

    count = (last_date_of_stock - first_date_of_stock).days

    valuation_panel = get_fundamentals_continuously(
        q, end_date=last_date_of_stock, count=count
    )
    if valuation_panel.empty:
        return
    for key in valuation_panel.minor_axis:
        df = valuation_panel.minor_xs(key)
        df = df.rename(columns={"code.1": "code", "day.1": "date"})
        stock_valuation_db.valuation.insert_many(df.to_dict("records"))
        # stock_valuation_df['day.1'].apply(update_stock_valuation,stock=key,df=stock_valuation_df)

    #        last_date_of_stock -= timedelta(days=500)
    print(
        "{0} {1} - {2} valuation done".format(
            stock, first_date_of_stock, last_date_of_stock
        )
    )


def get_stocks_value_by_date():
    stock_price_db = mc["stock_price"]
    stocks_code = stock_price_db.list_collection_names()
    stocks_code.sort()
    # get existing from stock_valuation_db
    # stock_valuation = mc["stock_valuation"]
    # date_list = stock_valuation.list_collection_names()
    # date_list.sort(reverse=True)
    # existing_stocks_code=pd.DataFrame(list(stock_valuation[date_list[0]].find({},{"code":1, "_id":0})))['code'][:-1]
    # stocks_code=list(set(stocks_code) - set(existing_stocks_code))
    stocks_series = pd.Series(stocks_code)
    stocks_series.apply(stock_value_by_date)


def get_stock_finforcast(stock):
    fundamental_db = mc["fundamental"]
    exist_df = pd.DataFrame(list(fundamental_db.finforcast.find({"code": stock})))
    if not exist_df.empty:
        print("fin forcast for {} exists".format(stock))
        return
    start_date, end_date = get_start_end_date(stock)
    q = query(finance.STK_FIN_FORCAST).filter(
        finance.STK_FIN_FORCAST.code == stock,
        finance.STK_FIN_FORCAST.pub_date <= end_date,
    )
    df = finance.run_query(q)
    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"], format=DATE_FORMAT)
        df["pub_date"] = pd.to_datetime(df["pub_date"], format=DATE_FORMAT)
        fundamental_db.finforcast.insert_many(df.to_dict("records"))
        print("Downloaded fin forcast for {}".format(stock))
    else:
        print("fin forcast for {} unavailable on JQDATA".format(stock))


@timing
def get_all_finforcast():
    all_securities_db = mc["all_securities"]
    all_securities = pd.DataFrame(list(all_securities_db.all_securities.find()))
    stocks = all_securities["index"]
    # change the number after '0' to specific scope
    stocks.apply(get_stock_finforcast)


def weekly_price(series, stock_df):
    print("1")


def take_first(array_like):
    if array_like.empty:
        return 0
    return array_like[0]


def take_last(array_like):
    if array_like.empty:
        return 0
    return array_like[-1]


def updateAll():
    Security().update()
    DailyPrice().updateAll()
    MoneyFlow().updateAll()
    IsST().updateAll()
    MTSS().updateAll()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")

    FUNCTION_MAP = {
        "updateSecurities": Security().update,
        "updatePriceD": DailyPrice().updateAll,
        "updatePriceW": WeeklyPrice().updateAll,
        "updatePriceM": MonthlyPrice().updateAll,
        "updateMoneyflow": MoneyFlow().updateAll,
        "updateSt": IsST().updateAll,
        "updateMtss": MTSS().updateAll,
        "updateFundQ": FundamentalQuarter().updateAll,
        "updateFundY": FundamentalYear().updateAll,
        "loadPrice": DailyPrice().loadAll,
        "adftest": algo().adftest,
        "hurst": algo().hurst,
        "cadftest": algo().cadftest,
        "updateIndicatorsD": indicators().update,
        "updateIndicatorsW": indicators(class_name="WeeklyPrice",db_name = "WeeklyIndicators").update,
        "updateIndicatorsM": indicators(class_name="MonthlyPrice",db_name = "MonthlyIndicators").update,
        # "updateUSPrice": UsPrice().updateAll,
        "stockcal": stockCal,
        "indexcal": indexCal,
        "industries": get_industries,
        "concepts": get_concepts,
        "industries_price": get_industries_price,
        "uptodate_industry_price": uptodate_industry_price,
        "stocks_value": get_stocks_value_by_date,
        "fin-forcast": get_all_finforcast,
        "convertstockprice": convertStockPrice,
        "convertindexprice": convertIndexPrice,
        "loadprice": loadprice,
        "updateAll": updateAll,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=FUNCTION_MAP.keys())

    args = parser.parse_args()

    JQData.connect()
    func = FUNCTION_MAP[args.command]

    func()

    # get_all_securities()
    # get_all_price()
    # get_all_industries()
    start = timeit.default_timer()
    # stockCal()
    # get_industries()
    # get_concepts()
    # get_industries_price()
    # uptodate_industry_price()
    # get_stocks_mtss_moneyflow()
    # get_stocks_value_orderby_date()
    # get_all_fundamentals()

    JQData.disconnect()


#                    fundamental_db.indicator.delete_one({"$and": [{"quarter_or_year": str(year)+'q'+str(i)},
#                                  {"code": stock}]})
