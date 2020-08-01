# encoding: UTF-8

from price import SecurityBase, DailyPrice, Valuation
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
from common import *
import jqdatasdk as jq
from jqdatasdk import *
import os


class Industry(object):
    def __init__(self):
        self.start_date = datetime.now().strftime(DATE_FORMAT)
        self.end_date = "2020-07-17"
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


class Concept(Industry):
    def __init__(self):
        super(Concept, self).__init__()
        self.db_name = "concept"
        self.db = self.db_conn[self.db_name]

    def update(self, concept_code, date):
        stock_list = get_concept_stocks(concept_code, date=date)
        self.df[concept_code] = pd.Series([stock_list])

    def updateAll(self):
        date = self.end_date
        # Get concept list
        concept_df = jq.get_concepts()
        concept_index = pd.Series(concept_df.index)
        concept_index.apply(self.update, date=date)
        self.db[date].insert_many(self.df.to_dict("records"))


class SW1DailyPrice(SecurityBase):
    def __init__(self, db_name="SW1DailyPrice"):
        super(SW1DailyPrice, self).__init__(db_name)
        self.fieldsFromDb = ["index"]
        self.index_column = "date"

    def query(self, index, start_date, end_date):
        q = query(finance.SW1_DAILY_PRICE).filter(
            finance.SW1_DAILY_PRICE.code == index,
            finance.SW1_DAILY_PRICE.date >= start_date,
            finance.SW1_DAILY_PRICE.date <= end_date,
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


class IndustryDailyPrice(SecurityBase):
    def __init__(self, db_name="IndustryDailyPrice"):
        super(IndustryDailyPrice, self).__init__(db_name)
        self.industryDailyPrice_df = pd.DataFrame()
        self.start_date = "2020-01-01"
        self.end_date = TODAY

    def calculate(self, series):
        industry_code = series.name

        # Select the industry stocks dataframe
        # df_dict = { index: self.dailyprice_df[index] for index in series[0] }
        industry_df = pd.DataFrame()
        for index in series[0]:
            if index not in self.valuation_df:
                continue
            index_df = self.valuation_df[index][["circulating_market_cap"]]
            index_df.columns = [index]
            if industry_df.empty:
                industry_df = index_df
            else:
                industry_df = industry_df.join(index_df, how="outer")
        if industry_df.empty:
            return
        # industry_df.iloc[[0]] = industry_df.iloc[0].fillna(0)
        industry_df.iloc[0, :] = industry_df.iloc[0].fillna(0)
        industry_df = industry_df.fillna(method="ffill")
        industry_df = industry_df[
            (industry_df.index >= self.start_date)
            & (industry_df.index <= self.end_date)
        ]
        industry_sum = industry_df.sum(axis=1)
        self.industryDailyPrice_df[industry_code] = industry_sum

    def updateAll(self):
        # Load all securities daily price
        self.dailyprice_df = DailyPrice().loadAll()
        # Load all securities valulation
        self.valuation_df = Valuation().loadAll()
        # Load industries stock list
        self.industries_df = Industry().loadAll()
        self.industries_df.apply(self.calculate)
        # save
        self.db["sum"].drop()
        self.industryDailyPrice_df["index"] = self.industryDailyPrice_df.index
        self.db["sum"].insert_many(self.industryDailyPrice_df.to_dict("records"))
        print("test")
