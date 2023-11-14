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


def date_enumerate(end_time):
    # code = "sh.000991"
    # code = 'HK.00700'
    # code = 'SH.513300'
    code = 'HK.09868'  # 小鹏
    begin_time = "2022-01-01"
    # data_src = DATA_SRC.BAO_STOCK
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_30M, KL_TYPE.K_5M]

    config = CChanConfig({
        "bi_strict": False,
        "bi_fx_check": "loss",
        "bi_algo": "normal",
        "bi_end_is_peak": False,
        "one_bi_zs": False,
        "triger_step": False,
        "skip_step": 0,
        # "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,1p,3a,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
        # "zs_algo": "over_seg"
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": False,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True,
        "plot_boll": True
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
            # "sub_lv_cnt": 6,
            # "facecolor": 'green',
            "plot_trendline": False
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
            "sub_lv_cnt": 30,
            "facecolor": 'green'
        },
        "figure": {
            "x_range": 10000,
            "only_top_lv": False
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }
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
    else:
        CAnimateDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
    # input()


if __name__ == '__main__':
    start = datetime.strptime("2023-11-13", "%Y-%m-%d")
    end = datetime.strptime("2023-11-13", "%Y-%m-%d")

    current_date = start

    while current_date <= end:
        current_date += timedelta(days=1)
        date_enumerate(current_date.strftime("%Y-%m-%d"))
