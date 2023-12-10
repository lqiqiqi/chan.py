#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 10:12
# @Author  : rockieluo
# @File    : YFinanceAPI.py


import yfinance as yf

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "Time": DATA_FIELD.FIELD_TIME,
        "Open": DATA_FIELD.FIELD_OPEN,
        "High": DATA_FIELD.FIELD_HIGH,
        "Low": DATA_FIELD.FIELD_LOW,
        "Close": DATA_FIELD.FIELD_CLOSE,
    }
    return [_dict[x] for x in fileds.split(",")]


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    return dict(zip(column_name, data))

def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
    if len(inp) == 10:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = minute = 0
    elif len(inp) == 17:
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    elif len(inp) == 19:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = int(inp[11:13])
        minute = int(inp[14:16])
    else:
        raise Exception(f"unknown time column from yfinance:{inp}")
    return CTime(year, month, day, hour, minute)


class YF(CCommonStockApi):
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(YF, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        autype_dict = {AUTYPE.QFQ: True, AUTYPE.HFQ: True, AUTYPE.NONE: False}
        fields = "Time,Open,High,Low,Close"
        rs = yf.download(
            tickers=self.code,
            start=self.begin_date,
            end=self.end_date,
            interval=self.__convert_type(),
            auto_adjust=autype_dict[self.autype],
        )
        if not len(rs):
            raise Exception("没有获取到数据")
        rs = rs.reset_index(drop=False, names='Time')
        rs['Time'] = rs['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        for i in range(len(rs)):
            yield CKLine_Unit(create_item_dict(rs.iloc[i, :].to_list(), GetColumnNameFromFieldList(fields)))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    def __convert_type(self):
        _dict = {
            KL_TYPE.K_DAY: '1d',
            KL_TYPE.K_WEEK: '1wk',
            KL_TYPE.K_MON: '1mo',
            KL_TYPE.K_1M: '1m',
            KL_TYPE.K_5M: '5m',
            KL_TYPE.K_15M: '15m',
            KL_TYPE.K_30M: '30m',
            KL_TYPE.K_60M: '60m',
        }
        return _dict[self.k_type]
