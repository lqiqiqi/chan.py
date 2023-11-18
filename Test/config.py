#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 11:06
# @Author  : rockieluo
# @File    : config.py
from Common.CEnum import KL_TYPE

chan_config = {
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
        "bs1_peak": True,
        "macd_algo": "peak",
        "bs_type": '1,2,1p,3a,2s,3b',
        "print_warning": True,
        # "zs_algo": "normal",
        "zs_algo": "over_seg"
}

plot_config = {KL_TYPE.K_DAY: {
    "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": True,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True,
        "plot_boll": True
    },
    KL_TYPE.K_30M: {
    "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True,
        "plot_boll": False
    },
    KL_TYPE.K_5M: {
    "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True,
        "plot_boll": False
},
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
        "sub_lv_cnt": 20,
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
