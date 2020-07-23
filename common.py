# encoding: UTF-8
import json
import dask
from datetime import datetime

config = open("config.json")
setting = json.load(config)

dask.config.set(scheduler="multiprocessing")


FIELDS = ["open", "high", "low", "close", "volume"]

STARTDATE = "2019-06-03"
ENDDATA = "2019-06-03"
DATE_FORMAT = "%Y-%m-%d"
JQDATA_STARTDATE = "2005-01-01"
JQDATA_ENDDATE = datetime.now().strftime(DATE_FORMAT)
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
