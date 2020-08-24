# encoding: UTF-8
from price import *
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import _pickle as pickle
from common import *
from sqlalchemy import and_
import os 
import csv
import requests
from zipfile import ZipFile
from io import BytesIO as Buffer
from io import (TextIOWrapper, BytesIO)
import yfinance as yf

ADDITIONAL_SECURITIES = ["^GSPC","^VIX","^TNX","^IRX"]


# These methods enable python 2 + 3 compatibility.
def get_zipfile_from_response(response):
    buffered = Buffer(response.content)
    return ZipFile(buffered)

def get_buffer_from_zipfile(zipfile, filename):
    return TextIOWrapper(BytesIO(zipfile.read(filename)))

class USSecurity(Security):
    def __init__(self):
        self.db_name = self.__class__.__name__
        self.cl_name = "securities"
        self.db_conn()
        
    def update(self):
        listing_file_url = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
        response = requests.get(listing_file_url)
        zipdata = get_zipfile_from_response(response)
        raw_csv = get_buffer_from_zipfile(zipdata, 'supported_tickers.csv')
        reader = csv.DictReader(raw_csv)
        rows = [row for row in reader
                if (row.get('assetType') == 'Stock' or row.get('assetType') == 'ETF') and
                row.get('priceCurrency') == 'USD' and row.get('endDate') != '' and (row.get('exchange') == 'NYSE'
                or row.get('exchange') == 'NASDAQ' or row.get('exchange') == 'AMEX' or
                row.get('exchange') == 'NYSE ARCA')]
        df = pd.DataFrame(rows)
        df.columns=['index','exchange','type','currency','start_date','end_date']
        df.drop_duplicates(subset=['index'], keep='first', inplace=True)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        df["type"] = df["type"].str.lower()
        
        #Manully add some securities.
        for s in ADDITIONAL_SECURITIES:
            df = df.append({'index': s, 
                            'exchange': 'NYSE', 
                            'type':'index', 
                            'currency':'USD',
                            'start_date': pd.to_datetime('1985-01-01'),
                            'end_date': pd.to_datetime(TODAY)}, ignore_index=True)
            
        self.cl.drop()
        self.cl.insert_many(df.to_dict("records"))
        print ("US security list updated")
        return 1    

class USDailyPrice(SecurityBase):
    def __init__(self, db_name="USDailyPrice"):
        super().__init__(db_name, USSecurity)

    def query(self, index, start_date, end_date):
        df = yf.download(index, 
                               start=start_date, 
                               end=end_date+timedelta(days=1),
                               progress=False)
        df=df.drop(columns=["Close"])
        df.columns = ["open","high","low","close","volume"]
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
        q = (
            query(finance.STK_XR_XD).filter
            (
                and_(
                    finance.STK_XR_XD.a_xr_date > start_date,
                    finance.STK_XR_XD.a_xr_date <= end_date
                )
            )
        )
        df = finance.run_query(q)  
        return df
    
    @JQData_decorate
    def updateAll(self):
        start_date = datetime.strptime("2010-01-01", DATE_FORMAT)
        if hasattr(self, "pickle_file") and os.path.exists(self.pickle_file):
            os.remove(self.pickle_file)

        try:
            #Get existing db for start_time
            self.df_dict["xr"] = pd.DataFrame(
                list(self.db["xr"].find({}, {"_id": 0}).sort("index", 1))
            ) 
            #Get max start_time
            start_date = self.df_dict["xr"].a_xr_date.max()
            start_date = datetime.strptime(start_date, DATE_FORMAT)
        except Exception:
            pass
        end_date = start_date + timedelta(days=100)
        if end_date > datetime.now():
            end_date = datetime.now()
        start_date = start_date.strftime(DATE_FORMAT)
        end_date = end_date.strftime(DATE_FORMAT)
        #Get new xr
        index = "xr"
        df = self.query(index, start_date, end_date)
        if not df.empty:
            df["code"].apply(self.updateOne)
            df["report_date"] = df.report_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["board_plan_pub_date"] = df.board_plan_pub_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["shareholders_plan_pub_date"] = df.shareholders_plan_pub_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["implementation_pub_date"] = df.implementation_pub_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["a_registration_date"] = df.a_registration_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_registration_date"] = df.b_registration_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["a_xr_date"] = df.a_xr_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_xr_baseday"] = df.b_xr_baseday.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_final_trade_date"] = df.b_final_trade_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["a_bonus_date"] = df.a_bonus_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_bonus_date"] = df.b_bonus_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["dividend_arrival_date"] = df.dividend_arrival_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_dividend_arrival_date"] = df.b_dividend_arrival_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["a_increment_listing_date"] = df.a_increment_listing_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_increment_listing_date"] = df.b_increment_listing_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["a_transfer_arrival_date"] = df.a_transfer_arrival_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
            df["b_transfer_arrival_date"] = df.b_transfer_arrival_date.apply(lambda x: x.strftime(DATE_FORMAT) if x is not None else None)
        self.db["xr"].insert_many(df.to_dict("records"))
        print(
            "{0} {1} to {2} {3} downloaded".format(
                index,
                start_date,
                end_date,
                self.__class__.__name__,
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
        stock_monthly_df = (
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
        stock_monthly_df = stock_monthly_df.dropna()
        stock_monthly_df["index"] = stock_monthly_df.index
        return stock_monthly_df

class Valuation(SecurityBase):
    def __init__(self):
        super(Valuation, self).__init__("Valuation")
        self.securityType = "stock"
        self.index_column = "day"
        #self.specify_date = True
        #self.start_date = datetime.strptime("2020-01-01", DATE_FORMAT)        
        #self.end_date = datetime.strptime("2020-06-30", DATE_FORMAT)          

    def query(self, index, start_date, end_date):
        #accumulative_df = pd.DataFrame()
        df = pd.DataFrame()
        #if start_date < datetime.strptime("2020-07-01", DATE_FORMAT):
        #if not index.startswith("6000"):
        #    return accumulative_df

        count = daycount(start_date, end_date)
        q = query(
            valuation
        ).filter(
            valuation.code.in_([index])
        )
        df = get_fundamentals_continuously(q, end_date=end_date, count=count) 
        df = df.drop(columns=["code.1"]).drop(columns=["day.1"])
        df = df.loc[(df["day"]>=start_date.strftime(DATE_FORMAT)) & 
                    (df["day"]<=end_date.strftime(DATE_FORMAT))]
        if not df.empty:
            df["day"] = pd.to_datetime(df["day"])    
        return df


#        for single_date in daterange(start_date, end_date):
#            q = query(
#                valuation
#            ).filter(
#                valuation.code == index
#            )
#            df = get_fundamentals(q, single_date.strftime(DATE_FORMAT)) 
#            if not df.empty:
#                accumulative_df = accumulative_df.append(df)
#        if not accumulative_df.empty:
#            accumulative_df["day"] = pd.to_datetime(accumulative_df["day"])    
#        return accumulative_df


class USCpi(Security):
    def __init__(self):
        self.db_name = self.__class__.__name__
        self.cl_name = "cpi"
        self.db_conn()    
        
    def update(self):
        import cpi
        cpi.update()
        print ("US CPI updated")
        return 1            
    
    