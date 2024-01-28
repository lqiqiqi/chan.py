#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 16:56
# @Author  : rockieluo
# @File    : TigerMockAPI.py
import os
from datetime import datetime

import pandas as pd
import pytz

from tigeropen.common.consts import BarPeriod
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else str2float(data[i])
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
    return CTime(year, month, day, hour, minute, auto=False)


def parse_time_column_to_timestamp(inp):
    # 根据输入字符串的长度解析年、月、日、时、分
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

    # 创建datetime对象
    dt = datetime(year, month, day, hour, minute)
    # 将datetime对象转换为美东时区
    et_timezone = pytz.timezone('US/Eastern')
    dt_et = et_timezone.localize(dt)
    # 将datetime对象转换为int格式的timestamp
    timestamp = int(dt_et.timestamp()) * 1000
    return timestamp


class TigerMock(CCommonStockApi):
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(TigerMock, self).__init__(code, k_type, begin_date, end_date, autype)
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段

    def get_kl_data(self, limit=1000):

        if self.begin_date is not None and self.begin_date != -1:
            self.begin_date = parse_time_column_to_timestamp(self.begin_date)
        else:
            self.begin_date = -1

        if self.end_date is not None and self.end_date != -1:
            self.end_date = parse_time_column_to_timestamp(self.end_date)
        else:
            self.end_date = -1

        rs = self.quote_client.get_future_bars(
            identifiers=[self.code],
            begin_time=self.begin_date,
            end_time=self.end_date,
            period=self.__convert_type(),
            limit=limit,
        )
        eastern = pytz.timezone('US/Eastern')

        rs['time'] += self.__offset_to_end_time()
        # 把时间偏移到每个bar的结束时间
        now_utc = datetime.now(pytz.utc).timestamp() * 1000
        # 如果是未走完的bar，丢弃
        rs = rs[rs['time'] < now_utc]
        rs['time_str'] = pd.to_datetime(rs['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
            eastern).dt.strftime('%Y-%m-%d %H:%M:%S')

        rs = rs.iloc[::-1][['time_str', 'open', 'high', 'low', 'close', 'volume']]
        for i in range(len(rs)):
            yield CKLine_Unit(create_item_dict(rs.iloc[i, :].to_list(), self.columns))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        file_path = f"{cur_path}/MockAccountConfig"
        cls.client_config = TigerOpenClientConfig(props_path=file_path)
        cls.quote_client = QuoteClient(cls.client_config)
        cls.trade_client = TradeClient(cls.client_config)

    @classmethod
    def do_close(cls):
        pass

    def __convert_type(self):
        _dict = {
            KL_TYPE.K_DAY: BarPeriod.DAY,
            KL_TYPE.K_WEEK: BarPeriod.WEEK,
            KL_TYPE.K_MON: BarPeriod.MONTH,
            KL_TYPE.K_1M: BarPeriod.ONE_MINUTE,
            KL_TYPE.K_5M: BarPeriod.FIVE_MINUTES,
            KL_TYPE.K_15M: BarPeriod.FIFTEEN_MINUTES,
            KL_TYPE.K_30M: BarPeriod.HALF_HOUR,
            KL_TYPE.K_60M: BarPeriod.ONE_HOUR,
        }
        return _dict[self.k_type]

    def __offset_to_end_time(self):
        # 老虎证券的每个bar的时间是开始时间，把它偏移到结束时间（天级别以上不是）
        _dict = {
            KL_TYPE.K_1M: 60 * 1000,
            KL_TYPE.K_5M: 5 * 60 * 1000,
            KL_TYPE.K_15M: 15 * 60 * 1000,
            KL_TYPE.K_30M: 30 * 60 * 1000,
            KL_TYPE.K_60M: 60 * 60 * 1000,
        }
        return _dict[self.k_type]