# encoding: UTF-8
from abc import ABCMeta, abstractmethod
from datetime import datetime
from datetime import timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
import pandas as pd
import numpy as np
import _pickle as pickle
import jqdatasdk as jq
from jqdatasdk import *
from common import *
from sqlalchemy import and_
import os


class Security(object):
    def __init__(self):
        self.db_name = "securities"
        self.cl_name = "securities"
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]
        self.cl = self.db[self.cl_name]
        # self.securities_df["index" == index]

    def update(self):
        self.cl.drop()
        securities_df = jq.get_all_securities(types=["stock", "fund", "index", "etf"])
        securities_df.reset_index(level=0, inplace=True)
        self.cl.insert_many(securities_df.to_dict("records"))
        print("security list updated")
        return 1

    def load(self, securityType: str = ""):
        find_string = {"type": securityType} if securityType != "" else {}
        return pd.DataFrame(list(self.cl.find(find_string)))

    def getSecurityDate(self, index):
        self.securities_df = self.load()
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
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db_name = db_name
        self.df_dict = {}
        self.securityType = ""
        self.index_column = ""
        self.fieldsFromDb = ["index"]
        self.pickle_file = self.db_name + ".pkl"
        self.specify_date = False
        self.db_connect()

    def db_connect(self):
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]

    def loadFieldsFromDB(self, fields=["index"]):
        self.security = Security()
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
        if self.specify_date is True:
            start_date = self.start_date
            end_date = self.end_date
        else:
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
        if hasattr(self, "pickle_file") and os.path.exists(self.pickle_file):
            os.remove(self.pickle_file)
        self.loadFieldsFromDB(fields=self.fieldsFromDb)
        security_df = self.security.load(securityType=self.securityType)
        security_df["index"].apply(self.updateOne)

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
                self.df_dict[security] = self.df_dict[security].fillna(0)
            fp = open(self.pickle_file, "wb")
            pickle.dump(self.df_dict, fp)

        return self.df_dict


class DailyPrice(SecurityBase):
    def __init__(self, db_name="DailyPrice"):
        super(DailyPrice, self).__init__(db_name)
        # self.pickle_file = "dailyprice.pkl"

    #    def updateOne(self, index):
    #        try:
    #            last_date = self.df_dict[index]["index"].iloc[-1].strftime(DATE_FORMAT)
    # If there are bonus or split after last_date,
    # need to re-download all price because of "pre" adjustment
    #            q = (
    #                query(finance.STK_XR_XD)
    #                .filter(
    #                    finance.STK_XR_XD.code == index,
    #                    finance.STK_XR_XD.a_bonus_date >= last_date,
    #                )
    #                .limit(100)
    #            )
    #            df = finance.run_query(q)
    #            if not df.empty:
    #                print("{0} has bonus after {1}".format(index, last_date))
    #                self.df_dict.pop(index, None)
    #                self.db[index].drop()
    #            else:
    #                q = (
    #                    query(finance.STK_XR_XD)
    #                    .filter(
    #                        finance.STK_XR_XD.code == index,
    #                        finance.STK_XR_XD.a_bonus_date >= last_date,
    #                    )
    #                    .limit(100)
    #                )
    #                df = finance.run_query(q)
    #                if not df.empty:
    #                    print("{0} has split after {1}".format(index, last_date))
    #                    self.df_dict.pop(index, None)
    #                    self.db[index].drop()

    #        except Exception:
    #            pass
    #        super(DailyPrice, self).updateOne(index)

    def query(self, index, start_date, end_date):
        df = jq.get_price(
            index, start_date, end_date, skip_paused=True, fill_paused=False, fq="pre"
        )
        return df


class SecurityAdj(SecurityBase):
    def __init__(self, db_name="SecurityAdj"):
        super(SecurityAdj, self).__init__(db_name)
        self.dailyprice_db = self.db_conn["DailyPrice"]
        self.df_dict = dict()

    def updateOne(self, index):
        try:
            self.dailyprice_db[index].drop()
        except Exception:
            pass

    def query(self, index, start_date, end_date):
        q = query(finance.STK_XR_XD).filter(
            and_(
                finance.STK_XR_XD.a_xr_date > start_date,
                finance.STK_XR_XD.a_xr_date <= end_date,
            )
        )
        df = finance.run_query(q)
        return df

    def updateAll(self):
        start_date = datetime.strptime("2010-01-01", DATE_FORMAT)
        if hasattr(self, "pickle_file") and os.path.exists(self.pickle_file):
            os.remove(self.pickle_file)

        try:
            # Get existing db for start_time
            self.df_dict["xr"] = pd.DataFrame(
                list(self.db["xr"].find({}, {"_id": 0}).sort("index", 1))
            )
            # Get max start_time
            start_date = self.df_dict["xr"].a_xr_date.max()
            start_date = datetime.strptime(start_date, DATE_FORMAT)
        except Exception:
            pass
        end_date = start_date + timedelta(days=100)
        if end_date > datetime.now():
            end_date = datetime.now()
        start_date = start_date.strftime(DATE_FORMAT)
        end_date = end_date.strftime(DATE_FORMAT)
        # Get new xr
        index = "xr"
        df = self.query(index, start_date, end_date)
        if not df.empty:
            df["code"].apply(self.updateOne)
            df["report_date"] = df.report_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["board_plan_pub_date"] = df.board_plan_pub_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["shareholders_plan_pub_date"] = df.shareholders_plan_pub_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["implementation_pub_date"] = df.implementation_pub_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["a_registration_date"] = df.a_registration_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_registration_date"] = df.b_registration_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["a_xr_date"] = df.a_xr_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_xr_baseday"] = df.b_xr_baseday.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_final_trade_date"] = df.b_final_trade_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["a_bonus_date"] = df.a_bonus_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_bonus_date"] = df.b_bonus_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["dividend_arrival_date"] = df.dividend_arrival_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_dividend_arrival_date"] = df.b_dividend_arrival_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["a_increment_listing_date"] = df.a_increment_listing_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_increment_listing_date"] = df.b_increment_listing_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["a_transfer_arrival_date"] = df.a_transfer_arrival_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
            df["b_transfer_arrival_date"] = df.b_transfer_arrival_date.apply(
                lambda x: x.strftime(DATE_FORMAT) if x is not None else None
            )
        self.db["xr"].insert_many(df.to_dict("records"))
        print(
            "{0} {1} to {2} {3} downloaded".format(
                index, start_date, end_date, self.__class__.__name__,
            )
        )


class WeeklyPrice(DailyPrice):
    def __init__(self, db_name="WeeklyPrice"):
        super(WeeklyPrice, self).__init__(db_name)
        self.pickle_file = "weeklyprice.pkl"
        self.dailyprice_df = {}

    def query(self, index, start_date, end_date):
        if not self.dailyprice_df:
            self.dailyprice_df = DailyPrice().loadAll()
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
                    "{0} {1} to {2} {3} calculated".format(
                        index,
                        start_date.strftime(DATE_FORMAT),
                        end_date.strftime(DATE_FORMAT),
                        self.__class__.__name__,
                    )
                )


class MonthlyPrice(WeeklyPrice):
    def __init__(self, db_name="MonthlyPrice"):
        super(MonthlyPrice, self).__init__(db_name)
        self.pickle_file = "monthlyprice.pkl"
        self.dailyprice_df = {}

    def query(self, index, start_date, end_date):
        if not self.dailyprice_df:
            self.dailyprice_df = DailyPrice().loadAll()
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


class Valuation(SecurityBase):
    def __init__(self):
        super(Valuation, self).__init__("Valuation")
        self.securityType = "stock"
        self.index_column = "day"
        self.specify_date = True
        self.start_date = datetime.strptime("2020-01-01", DATE_FORMAT)
        self.end_date = datetime.strptime("2020-06-30", DATE_FORMAT)

    def query(self, index, start_date, end_date):
        accumulative_df = pd.DataFrame()
        # if start_date < datetime.strptime("2020-07-01", DATE_FORMAT):

        for single_date in daterange(start_date, end_date):
            q = query(valuation).filter(valuation.code == index)
            df = get_fundamentals(q, single_date.strftime(DATE_FORMAT))
            if not df.empty:
                accumulative_df = accumulative_df.append(df)
        if not accumulative_df.empty:
            accumulative_df["day"] = pd.to_datetime(accumulative_df["day"])
        return accumulative_df
