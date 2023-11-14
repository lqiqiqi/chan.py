#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 09:00
# @Author  : rockieluo
# @File    : FuTuAPI.py


from datetime import datetime

from futu import OpenQuoteContext, AuType, KL_FIELD, KLType

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time_key": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
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
        raise Exception(f"unknown time column from futu:{inp}")
    return CTime(year, month, day, hour, minute)


class Futu(CCommonStockApi):
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(Futu, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        fields = "time_key,open,high,low,close"

        autype_dict = {AUTYPE.QFQ: AuType.QFQ, AUTYPE.HFQ: AuType.HFQ, AUTYPE.NONE: AuType.NONE}
        field_list = [KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH, KL_FIELD.LOW, KL_FIELD.CLOSE]
        rs = self.quote_ctx.request_history_kline(
            code=self.code,
            start=self.begin_date,
            end=self.end_date,
            fields=field_list,
            ktype=self.__convert_type(),
            autype=autype_dict[self.autype],
            max_count=1000000,
        )
        if rs[0] != 0:
            raise Exception(rs[1])
        for i in range(len(rs[1])):
            yield CKLine_Unit(create_item_dict(rs[1].iloc[i, 2:].to_list(), GetColumnNameFromFieldList(fields)))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        cls.quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)  # 创建行情对象

    @classmethod
    def do_close(cls):
        cls.quote_ctx.close()

    def __convert_type(self):
        _dict = {
            KL_TYPE.K_DAY: KLType.K_DAY,
            KL_TYPE.K_WEEK: KLType.K_WEEK,
            KL_TYPE.K_MON: KLType.K_MON,
            KL_TYPE.K_5M: KLType.K_5M,
            KL_TYPE.K_15M: KLType.K_15M,
            KL_TYPE.K_30M: KLType.K_30M,
            KL_TYPE.K_60M: KLType.K_60M,
        }
        return _dict[self.k_type]
