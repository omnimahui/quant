# encoding: UTF-8
#from __future__ import print_function

from price import *
from industry import *
from indicators import *
from finance import *
from common import *
from algo import *
from USprice import *



from abc import ABCMeta, abstractmethod
import sys
import json
from datetime import datetime
from datetime import timedelta
from time import time, sleep

from pymongo import MongoClient, ASCENDING, DESCENDING
import pandas as pd
import numpy as np

#from vnpy.trader.object import BarData, TickData

# from vnpy.trader.app.ctaStrategy.ctaBase import (MINUTE_DB_NAME,
#                                                 DAILY_DB_NAME,
#                                                 TICK_DB_NAME,
#                                                 ETF_DAILY_DB_NAME)
import jqdatasdk as jq
from jqdatasdk import *
import multiprocessing
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


import math
import os
from common import *


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
    SecurityAdj().updateAll
    Security().update()
    DailyPrice().updateAll()
    MoneyFlow().updateAll()
    IsST().updateAll()
    MTSS().updateAll()
    Valuation().updateAll()
    WeeklyPrice().updateAll()
    MonthlyPrice().updateAll()
    IndustryDailyPrice().updateAll
    indicators().update()
    indicators(class_name="WeeklyPrice", db_name="WeeklyIndicators").update()
    indicators(class_name="MonthlyPrice", db_name="MonthlyIndicators").update()

def updateAllUS():
    USSecurity().update()
    USDailyPrice().updateAll()
    

def test():
    q = (
        query(finance.STK_XR_XD)
        .filter(
            finance.STK_XR_XD.a_xr_date >= "2020-01-01",
            finance.STK_XR_XD.code == "601066.XSHG",
        )
        .limit(10)
    )
    df = finance.run_query(q)
    print(df)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")

    FUNCTION_MAP = {
        "updateSecurityAdjustment": SecurityAdj().updateAll,
        "updateSecurities": Security().update,
        "updateUSSecurities": USSecurity().update,
        "updatePriceD": DailyPrice().updateAll,
        "updateUSPriceD": USDailyPrice().updateAll,
        "updatePriceW": WeeklyPrice().updateAll,
        "updatePriceM": MonthlyPrice().updateAll,
        "updateIndustry": Industry().updateAll,
        "updateMoneyflow": MoneyFlow().updateAll,
        "updateSt": IsST().updateAll,
        "updateMtss": MTSS().updateAll,
        "updateFundQ": FundamentalQuarter().updateAll,
        "updateFundY": FundamentalYear().updateAll,
        "updateIncomeQ": IncomeQuarter().updateAll,
        "updateBalanceQ": BalanceQuarter().updateAll,
        "updateValuation": Valuation().updateAll,
        "updateSW1": SW1DailyPrice().updateAll,
        "loadPrice": DailyPrice().loadAll,
        "adf": algo().adf,
        "hurst": algo().hurst,
        "cadftest": algo().cadftest,
        "percent": algo().percent,
        "ebittevroc": algo().ebittev,
        "test": IncomeQuarter().loadAll,
        "kalman": algo().kalman,
        "johansen": algo().johansen,
        "index_vs_stocks": algo().index_vs_stocks,
        "longShortStocks": algo().longShortStocks,
        "autoCorrel": algo().autoCorrel,''
        "momentum_top": algo().momentum_top,
        "kelly": algo().kelly,
        "volatility": algo().volatility,
        "pdf": algo().pdf,
        "acf": algo().acf,
        "correl": algo().correl,
        "correlSearch": algo().correlSearch,
        "priceChart": algo().priceChart,
        "decompose": algo().decompose,
        
        "updateIndicatorsD": indicators().update,
        "updateIndicatorsW": indicators(
            class_name="WeeklyPrice", db_name="WeeklyIndicators"
        ).update,
        "updateIndicatorsM": indicators(
            class_name="MonthlyPrice", db_name="MonthlyIndicators"
        ).update,
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
        "updateAllUS": updateAllUS,
        "updateIndustryPriceD": IndustryDailyPrice().updateAll,
        "updateConcept": Concept().updateAll
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=FUNCTION_MAP.keys())

    args = parser.parse_args()

    #JQData.connect()
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

    #JQData.disconnect()


#                    fundamental_db.indicator.delete_one({"$and": [{"quarter_or_year": str(year)+'q'+str(i)},
#                                  {"code": stock}]})
