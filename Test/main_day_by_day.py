#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 09:50
# @Author  : rockieluo
# @File    : main_day_by_day.py
from datetime import datetime, timedelta

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver
from config import Config, plot_para, plot_config


def date_enumerate(end_time):
    # code = "sh.000991"
    code = 'HK.00700'
    # code = 'SH.513300'
    # code = 'HK.800700'
    # code = 'QQQ'
    end_time_dt = datetime.strptime(end_time, '%Y-%m-%d')
    begin_time = (end_time_dt - timedelta(days=60)).strftime('%Y-%m-%d')
    # data_src = DATA_SRC.BAO_STOCK
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]

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

    if not config.triger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        # plot_driver.figure.show()
        plot_driver.figure.savefig(f'../TestImage/{code.split(".")[-1]}/day/{end_time}.jpg')

    # input()


if __name__ == '__main__':
    start = datetime.strptime("2023-11-04", "%Y-%m-%d")
    end = datetime.strptime("2023-12-12", "%Y-%m-%d")

    current_date = start

    while current_date <= end:
        current_date += timedelta(days=1)
        try:
            date_enumerate(current_date.strftime("%Y-%m-%d"))
        except:
            print(current_date, '拉取失败')
