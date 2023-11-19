#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 11:26
# @Author  : rockieluo
# @File    : check_is_1_3_bp.py

from datetime import timedelta, datetime

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE

from Test.config import Config


def check_is_1_3_bsp(code, date):

    date_format = "%Y-%m-%d"
    date_object = datetime.strptime(date, date_format)
    begin_date = (date_object - timedelta(days=60)).strftime('%Y-%m-%d')

    begin_time = begin_date
    end_time = date
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_5M]
    config_object = Config()
    chan_config = config_object.read_chan_config
    config = CChanConfig(chan_config)

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    bsp_list = chan.get_seg_bsp(idx=0)
    bsp = bsp_list[-1]
    if (BSP_TYPE.T1 in bsp.type or BSP_TYPE.T3A in bsp.type) and bsp.is_buy:
        return True
    else:
        return False


