# encoding: UTF-8

from price import SecurityBase
from jqdatasdk import *
from datetime import datetime, timedelta
from common import *
import numpy as np
import pandas as pd


class FundamentalQuarter(SecurityBase):
    def __init__(self, db_name="FundamentalQuarter"):
        super().__init__(db_name)
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


        try:        
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
        except Exception as ex:
            print ("Got exception index={} {}".format(index, ex))

class IncomeQuarter(FundamentalQuarter):
    def __init__(self, db_name="IncomeQuarter"):
        super().__init__(db_name)
        self.securityType = "stock"
        self.index_column = "end_date"
        self.fieldsFromDb = ["index", "end_date"]
        self.start_from = "2004-09-30"

    def query(self, index, start_date, end_date):
        q = (
            query(finance.STK_INCOME_STATEMENT)
            .filter(
                finance.STK_INCOME_STATEMENT.code == index,
                finance.STK_INCOME_STATEMENT.end_date >= start_date,
                finance.STK_INCOME_STATEMENT.report_type == 0,
            )
            .limit(3000)
        )
        df = finance.run_query(q)
        if not df.empty:
            df = df.drop(columns=["id"])
            # Change str to datetime
            df["end_date"] = df.end_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["start_date"] = df.start_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["pub_date"] = df.pub_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["report_date"] = df.report_date.apply(lambda x: x.strftime(DATE_FORMAT))
        return df

    def updateOne(self, index):
        start_date, end_date = self.security.getSecurityDate(index)
        try:
            last_date = self.df_dict[index]["index"].iloc[-1]
            last_end_date = (
                self.df_dict[index]
                .loc[self.df_dict[index]["index"] == last_date]["end_date"]
                .values[0]
            )
            last_end_date = datetime.strptime(last_end_date, DATE_FORMAT)
            # Timestamp to datetime
            start_date = last_end_date + timedelta(days=1)
        except Exception:
            ...

        if start_date <= end_date:
            # Get end_date > start_date income report
            df = self.query(index, start_date, end_date)
            if not df.empty:
                self.set_index_column(df, self.index_column)
                self.db[index].insert_many(df.to_dict("records"))


class BalanceQuarter(IncomeQuarter):
    def __init__(self, db_name="BalanceQuarter"):
        super().__init__(db_name)
        self.securityType = "stock"
        self.index_column = "end_date"
        self.fieldsFromDb = ["index", "end_date"]
        self.start_from = "2004-09-30"
        
    def query(self, index, start_date, end_date):
        q = (
            query(finance.STK_BALANCE_SHEET)
            .filter(
                finance.STK_BALANCE_SHEET.code == index,
                finance.STK_BALANCE_SHEET.end_date >= start_date,
                finance.STK_BALANCE_SHEET.report_type == 0,
            )
            .limit(3000)
        )
        df = finance.run_query(q)
        if not df.empty:
            df = df.drop(columns=["id"])
            # Change str to datetime
            df["end_date"] = df.end_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["start_date"] = df.start_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["pub_date"] = df.pub_date.apply(lambda x: x.strftime(DATE_FORMAT))
            df["report_date"] = df.report_date.apply(lambda x: x.strftime(DATE_FORMAT))
        return df


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
