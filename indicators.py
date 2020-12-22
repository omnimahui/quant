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
import pandas_ta as ta


        


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
    
    def rsi(self, index, window=4):
        ts = self.df_dict[index].close
        rsi = ta.rsi(ts, length=window)
        rsi.name = "rsi_"+str(window)
        return rsi

    def rsi_streak(self, index, window=2):
        ts = self.df_dict[index].close
        p = ((ts-ts.shift(1)).fillna(0) > 0).astype(int)
        n = ((ts-ts.shift(1)).fillna(0) < 0).astype(int).replace(1,-1)
        streak = p.add(n)
        streak=streak.groupby((streak != streak.shift()).cumsum()).cumsum()
        rsi_streak=ta.rsi(streak, window)
        return rsi_streak        

    def percentile(self, pct_ts):
        percentile = pct_ts[pct_ts < pct_ts.iloc[-1]].count()/(pct_ts.count()-1)
        return percentile
        
    def pctRank(self, index, lookback=100):
        ts = self.df_dict[index].close
        pct_ts = ts.pct_change()
        pctRank = pct_ts.rolling(window=lookback+1, center=False).\
                  apply(self.percentile, raw=False).fillna(0)
        return pctRank*100
        
    
    def crsi(self, index, window = 100):
        rsi_3 = self.rsi(index, window=3)
        rsi_streak = self.rsi_streak(index, window=2)
        pctRank = self.pctRank(index, window)
        return (rsi_3+rsi_streak+pctRank)/3
        

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
                rsi_4 = self.rsi(security, 4)
                rsi_8 = self.rsi(security, 8)
                rsi_14 = self.rsi(security, 14)
                crsi = self.crsi(security, 100)
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
                    rsi_4,
                    rsi_8,
                    rsi_14,
                    crsi,
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