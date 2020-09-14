# encoding: UTF-8

from pymongo import MongoClient, ASCENDING, DESCENDING
from price import SecurityBase, DailyPrice, WeeklyPrice, MonthlyPrice
import jqdatasdk as jq
from jqdatasdk import *
from datetime import datetime, timedelta
from common import *
import os
import pandas as pd
import _pickle as pickle


        


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

class indicators(object):
    def __init__(self, class_name="DailyPrice", db_name="DailyIndicators"):
        # Load price timeseries
        #
        self.db_name = db_name
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]
        self.pickle_file = self.db_name + ".pkl"
        self.price_class = class_name

    def sma(self, index, window):
        ts = self.df_dict[index].close
        sma = ts.rolling(window=window, center=False).mean()
        sma.name = "sma_" + str(window)
        # sma = sma.reindex(ts.index)
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
        # atr=atr.reindex(self.df_dict[index].index)
        return atr

    def ewma(self, ts, window=9):
        ewmaSeries = ts.ewm(span=window, adjust=False).mean()
        return ewmaSeries

    def dif(self, index, fast=12, slow=26):
        ts = self.df_dict[index].close
        dif = self.ewma(ts, window=fast) - self.ewma(ts, window=slow)
        dif.name = "dif"
        return dif

    def dea(self, dif, window=9):
        dea = self.ewma(dif, window=window)
        dea.name = "dea"
        return dea

    def macd(self, dif, dea):
        macd = (dif - dea) * 2
        macd.name = "macd"
        return macd

    def update(self):
        # For each index
        # Calculate all indicaters
        self.df_dict = eval(self.price_class)().loadAll()
        if hasattr(self, "pickle_file") and os.path.exists(self.pickle_file):
            os.remove(self.pickle_file)
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
                # Insert into DB
                self.db[security].drop()
                self.db[security].insert_many(df.to_dict("records"))

            except Exception as ex:
                print("Got exception: {0} {1}".format(security, ex))

    def loadAll(self):
        try:
            fp = open(self.pickle_file, "rb")
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
            fp = open(self.pickle_file, "wb")
            pickle.dump(self.df_dict, fp)
        return self.df_dict