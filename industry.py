# encoding: UTF-8

from price import SecurityBase
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
from common import *
import jqdatasdk as jq
from jqdatasdk import *


class Industry(object):
    def __init__(self):
        self.start_date = datetime.now().strftime(DATE_FORMAT)
        self.end_date = "2020-07-17"  # datetime.now().strftime(DATE_FORMAT)
        self.db_name = "industry"
        self.db_conn = MongoClient(host=MONGODB_HOST)
        self.db = self.db_conn[self.db_name]
        self.df = pd.DataFrame()

    def update(self, industry_code, date):
        stock_list = get_industry_stocks(industry_code, date=date)
        self.df[industry_code] = pd.Series([stock_list])

    def updateAll(self):
        date = self.end_date

        for industry_category in ["zjw", "sw_l1", "sw_l2", "sw_l3", "jq_l1", "jq_l2"]:
            category_df = jq.get_industries(name=industry_category, date=date)
            category_index = pd.Series(category_df.index)
            category_index.apply(self.update, date=date)

        self.db[date].insert_many(self.df.to_dict("records"))

    def loadAll(self):
        df = pd.DataFrame(
            list(self.db[self.end_date].find({}, {"_id": 0}).sort("index", 1))
        )
        return df


class SW1DailyPrice(SecurityBase):
    def __init__(self, db_name="SW1DailyPrice"):
        super(SW1DailyPrice, self).__init__(db_name)
        self.fieldsFromDb = ["index"]
        self.index_column = "date"

    def query(self, index, start_date, end_date):
        q = query(finance.SW1_DAILY_PRICE).filter(
            finance.SW1_DAILY_PRICE.code == index,
            finance.SW1_DAILY_PRICE.date > start_date,
            finance.SW1_DAILY_PRICE.date < end_date,
        )
        df = finance.run_query(q)
        if not df.empty:
            df = df.drop(columns=["id"])
        return df

    def updateOne(self, index: str):
        start_date = datetime.strptime("2005-01-01", DATE_FORMAT)
        end_date = datetime.strptime(JQDATA_ENDDATE, DATE_FORMAT)
        try:
            last_date = self.df_dict[index]["index"].iloc[-1]
            # Timestamp to datetime
            start_date = (last_date + timedelta(days=1)).to_pydatetime()
        except Exception:
            ...

        if start_date <= end_date:
            df = self.query(index, start_date, end_date)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
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
        # get all SW1 industries
        industries_df = jq.get_industries(name="sw_l1", date=JQDATA_ENDDATE)
        industries_df.index.map(self.updateOne)
