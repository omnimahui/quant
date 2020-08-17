# encoding: UTF-8
import json
import dask
from datetime import datetime, timedelta
import jqdatasdk as jq
import numpy as np
import re

config = open("config.json")
setting = json.load(config)

dask.config.set(scheduler="multiprocessing")


FIELDS = ["open", "high", "low", "close", "volume"]

STARTDATE = "2019-06-03"
ENDDATA = "2019-06-03"
DATE_FORMAT = "%Y-%m-%d"
JQDATA_STARTDATE = "2005-01-01"
JQDATA_ENDDATE = datetime.now().strftime(DATE_FORMAT)
TODAY=datetime.now().strftime(DATE_FORMAT)
DASK_NPARTITIONS = 50
MONGODB_HOST = "127.0.0.1"


class JQData(object):
    @staticmethod
    def connect():
        if jq.is_auth() == False:
            setting = json.load(open("config.json"))
            jq.auth(setting["jqUsername"], setting["jqPassword"])

    @staticmethod
    def disconnect():
        jq.logout()

def JQData_decorate(func):
    def func_wrapper(*args, **kwargs):
        JQData.connect()
        func(*args, **kwargs)
        JQData.disconnect()
    return func_wrapper
        
def daterange(start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, DATE_FORMAT)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, DATE_FORMAT)
    #if start_date < datetime.strptime("2020-07-15", DATE_FORMAT):
    #    start_date = datetime.strptime("2020-07-15", DATE_FORMAT)
    #start_date = datetime.strptime("2020-07-01", DATE_FORMAT)
    #end_date = datetime.strptime("2020-07-24", DATE_FORMAT)
    for n in range(int((end_date - start_date).days + 1)):
        yield start_date + timedelta(n)  
        
def daycount(start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, DATE_FORMAT)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, DATE_FORMAT)
    count = int((end_date - start_date).days + 1)
    return count


def take_first(array_like):
    if array_like.empty:
        return 0
    return array_like[0]


def take_last(array_like):
    if array_like.empty:
        return 0
    return array_like[-1]

def calculateMaxDD(cumret):
# =============================================================================
# calculation of maximum drawdown and maximum drawdown duration based on
# cumulative COMPOUNDED returns. cumret must be a compounded cumulative return.
# i is the index of the day with maxDD.
# =============================================================================
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    drawdownduration=np.zeros(cumret.shape)
    
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=(1+cumret[t])/(1+highwatermark[t])-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
             
    maxDD, i=np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
    maxDDD=np.max(drawdownduration)
    return maxDD, maxDDD, i

def isChinaMkt(index):
    if re.match("^[0-9]+", index):
        return True
    else:
        return False
        