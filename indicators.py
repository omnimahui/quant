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
from sklearn.linear_model import LinearRegression
import math
import multiprocessing

        


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
        crsi = (rsi_3+rsi_streak+pctRank)/3
        crsi.name = "crsi"
        return crsi

    def momentum(self, index, window = 20):
        ts = self.df_dict[index].close
        adj_slope = ts.rolling(window=window, center=False).apply(
            self.momentumInWindow, raw=False
        )
        adj_slope.name = "momentum_" + str(window)
        adj_slope = adj_slope.reindex(ts.index)
        return adj_slope

    def momentumInWindow(self, ts):
        logArray = np.log(ts.values)
        x = pd.Series(np.arange(logArray.size)).to_numpy().reshape(-1, 1)
        logArray = logArray.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x, logArray)  # perform linear regression
        slope = linear_regressor.coef_
        r2 = linear_regressor.score(x, logArray)
        annual_slope = math.pow(np.exp(slope), 250) - 1
        adj_slope = annual_slope * r2
        return adj_slope    
           

    def update(self):
        # For each index
        # Calculate all indicaters
        self.df_dict = eval(self.price_class)().loadAll()
        if hasattr(self, "pickle_file") and os.path.exists(self.pickle_file):
            os.remove(self.pickle_file)
        num_worker = 8#multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        worker_queue = manager.Queue()
        for security in self.df_dict.keys():
            worker_queue.put(security)
        pool = multiprocessing.Pool(num_worker)
        for i in range(num_worker):
            pool.apply_async(worker, args=(worker_queue, i, self.db_name, self.price_class))
        # stop workers
        for i in range(num_worker):
            worker_queue.put(None)
    
        pool.close()
        pool.join()            
            

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
    

def worker(queue, worker_id, db_name, class_name):
    instance = indicators(class_name=class_name, db_name=db_name)
    instance.df_dict = eval(instance.price_class)().loadAll()
    while True:
        security = queue.get()
        if security is None:
            break            
        try:
            sma_21 = instance.sma(security, 21)
            sma_55 = instance.sma(security, 55)
            sma_200 = instance.sma(security, 200)
            rsi_2 = instance.rsi(security, 2)
            rsi_4 = instance.rsi(security, 4)
            rsi_8 = instance.rsi(security, 8)
            rsi_14 = instance.rsi(security, 14)
            crsi = instance.crsi(security, 100)
            atr_20 = instance.atr(security, 20)
            atr_60 = instance.atr(security, 60)
            atr_100 = instance.atr(security, 100)
            dif = instance.dif(security)
            dea = instance.dea(dif)
            macd = instance.macd(dif, dea)
            momentum_20 = instance.momentum(security, 20)
            momentum_60 = instance.momentum(security, 60)
            momentum_120 = instance.momentum(security, 120)
            momentum_250 = instance.momentum(security, 250)
            df = pd.DataFrame(index=instance.df_dict[security].index)
            for series in [
                sma_21,
                sma_55,
                sma_200,
                atr_20,
                atr_60,
                atr_100,
                dif,
                dea,
                macd,
                momentum_20,
                momentum_60,
                momentum_120,
                momentum_250,
                rsi_2,
                rsi_4,
                rsi_8,
                rsi_14,
                crsi,
            ]:
                df = df.merge(series.to_frame(), left_index=True, right_index=True)

            df["index"] = instance.df_dict[security].index
            # Insert into DB
            instance.db[security].drop()
            instance.db[security].insert_many(df.fillna(0).to_dict("records"))
            print("{0} indicators done in worker {1}".format(security, worker_id))

        except Exception as ex:
            print("Got exception: {0} {1}".format(security, ex))   
    queue.task_done()
    return 0         