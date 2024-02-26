# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 15:50
# @Author  : rockieluo
# @File    : buy_data_generation.py


import json
import math
import dill
from joblib import load
from typing import Dict, TypedDict
import sys

sys.path.append('/root/chan.py')


from DataAPI.TigerMockAPI import TigerMock
from DataAPI.YFinanceAPI import YF
from DataAPI.csvAPI import CSV_API

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from Test.config import Config
from candlestick import candlestick



def cal_atr(cur_lv_chan, n=14):
    high = np.array([klu.high for ckl in cur_lv_chan[-100:] for klu in ckl])
    low = np.array([klu.low for ckl in cur_lv_chan[-100:] for klu in ckl])
    close = np.array([klu.close for ckl in cur_lv_chan[-100:] for klu in ckl])
    atr_sum = 0
    for i in range(n, 0, -1):
        tr = max(high[-i] - low[-i], abs(high[-i] - close[-(i + 1)]), abs(low[-i] - close[-(i + 1)]))
        atr_sum += tr

    atr = atr_sum / n
    return atr


def train_buy_model(code, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    data_src = DATA_SRC.CSV
    # data_src = DATA_SRC.TigerMock
    lv_list = [KL_TYPE.K_DAY]

    config_object = Config()
    chan_config = config_object.read_chan_config_trigger_step
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

    data_src_5m = YF(code, k_type=KL_TYPE.K_DAY, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)

    last_break = None
    last_break_bsp = None
    last_break_zs_begin_time = None
    last_break_zs_end_time = None
    last_break_zs_high = None
    first_indx = 0
    is_hold = False

    trade_info = {'code': [], 'buy_time': [], 'buy_price': [], 'sell_time': [], 'sell_price': [],
                  'profit': [], 'real_profit': [], 'sell_reason': []
                  }
    for last_5m_klu in data_src_5m.get_kl_data():
        chan.trigger_load({KL_TYPE.K_DAY: [last_5m_klu]})
        if first_indx < 365:
            first_indx += 1
            continue
        chan_snapshot = chan
        bsp_list = chan_snapshot.get_bsp()
        seg_bsp_list = chan_snapshot.get_seg_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        bp_list = [bsp for bsp in bsp_list if bsp.is_buy]
        if not bp_list:
            continue
        last_bp = bp_list[-1]
        # if last_bsp.is_buy:
        # print(last_bsp.klu.time)

        cur_lv_chan = chan_snapshot[0]
        last_zs = cur_lv_chan.zs_list[-1]
        last_zs_high = max([ckl.high for bi in last_zs.bi_lst for ckl in bi.klc_lst])
        last_last_5m_klu = [klu for ckl in cur_lv_chan[-2:] for klu in ckl][-2]
        last_last_last_5m_klu = [klu for ckl in cur_lv_chan[-3:] for klu in ckl][-3]
        atr = cal_atr(cur_lv_chan)
        last_bi = cur_lv_chan.bi_list[-1]
        last_bi_highest_close = max([klu.close for klc in last_bi.klc_lst for klu in klc])
        if len(cur_lv_chan.seg_list) < 2:
            return None
        last_last_seg = cur_lv_chan.seg_list[-2]

        if is_hold is False and (last_last_5m_klu.close < last_zs_high + atr < last_5m_klu.close or
                 last_last_last_5m_klu.close < last_zs_high + atr < last_last_5m_klu.close) \
                and not (BSP_TYPE.T1P in last_bp.type or BSP_TYPE.T1 in last_bp.type) and \
                last_5m_klu.close >= last_bi_highest_close:
            if last_zs.begin_bi.get_begin_klu().time < last_last_seg.get_end_klu().time and not last_last_seg.is_up():
                continue
            bi_distance = last_bi.idx - last_zs.end_bi.idx
            if bi_distance > 2:
                continue

            last_buy_price = last_5m_klu.close
            last_buy_time = last_5m_klu.time
            last_buy_zs_high = last_zs_high
            print(f'{last_buy_time}: buy price = {last_buy_price} ')
            print('bp time', last_bp.klu.time, 'klu_time', last_5m_klu.time.ts, last_5m_klu.time.to_str(), 'last_zs.begin.time', last_zs.begin.time, 'last_zs.end.time', last_zs.end.time, 'last_zs high', last_zs_high, 'last_bsp_time', last_bp.klu.time, 'last_bsp is buy', last_bp.is_buy)
            is_hold = True
            trade_info['code'] = code
            trade_info['buy_time'].append(last_5m_klu.time.to_str())
            trade_info['buy_price'].append(last_buy_price)

            last_break = last_5m_klu.time.to_str()
            last_break_bsp = last_bp.klu.time
            last_break_zs_begin_time = last_zs.begin.time
            last_break_zs_end_time = last_zs.end.time
            last_break_zs_high = last_zs_high

        if is_hold:
            # if last_5m_klu.close < last_buy_zs_high:
            #     is_hold = False
            #     sell_price = last_5m_klu.close
            #     sell_reason = 'lt last_buy_zs_high'
            #     print(f'{last_5m_klu.time}: lt last_buy_zs_high sell price = {sell_price} ')
            if last_last_5m_klu.trend[TREND_TYPE.MEAN][5] > last_last_5m_klu.trend[TREND_TYPE.MEAN][10] and \
                last_5m_klu.trend[TREND_TYPE.MEAN][5] < last_5m_klu.trend[TREND_TYPE.MEAN][10]:
                sell_price = last_5m_klu.close
                sell_reason = 'dead cross'
                is_hold = False
            # elif last_bp.is_buy is False and cur_lv_chan[-3].idx == last_bp.klu.klc.idx and last_bp.klu.close > last_buy_price:
            #     sell_price = last_5m_klu.close
            #     sell_reason = 's1p'
            #     is_hold = False
            # elif last_bp.is_buy is False and cur_lv_chan[-2].idx == last_bp.klu.klc.idx and (last_5m_klu.close - last_5m_klu.open)/last_5m_klu.open < -0.0005:
            #     sell_price = last_5m_klu.close
            #     sell_reason = 's1p and one klu retrace gt 0.0005'
            #     is_hold = False

            if not is_hold:
                print('sell at ', last_5m_klu.time.to_str())
                trade_info['sell_reason'].append(sell_reason)
                trade_info['sell_time'].append(last_5m_klu.time.to_str())
                trade_info['sell_price'].append(sell_price)
                trade_info['profit'].append((sell_price - last_buy_price) / last_buy_price * 100)
                trade_info['real_profit'].append((sell_price - last_buy_price))

    trade_df = pd.DataFrame(trade_info)
    return last_break, last_break_bsp, last_break_zs_begin_time, last_break_zs_end_time, last_break_zs_high, trade_df


if __name__ == '__main__':
    res_dict = {'code': [], 'last_break': [],  'last_break_bsp': [], 'last_break_zs_begin_time': [],
                'last_break_zs_end_time': [], 'last_break_zs_high': []}
    all_code_df = pd.DataFrame()
    for code in ['VTI', 'OEF', 'SPY', 'DIA', 'MDY', 'RSP', 'QQQ', 'QTEC', 'IWB', 'IWM', 'MTUM', 'SPHB',
                 'QUAL', 'SPLV', 'RSPC', 'RSPD', 'RSPS', 'RSPG', 'RSPF', 'RSPH', 'RSPN', 'RSPM', 'RSPR', 'RSPT',
                 'RSPU', 'IWY', 'IVW', 'IWF', 'IWO', 'METV', 'IPO', 'SNSR', 'XT', 'MOAT', 'SOCL', 'ONLN', 'SKYY',
                 'HERO', 'IBUY', 'IPAY', 'FINX', 'CIBR', 'IGF', 'DRIV', 'BOTZ', 'ROBO', 'MOO', 'TAN', 'QCLN', 'PBW']:

        try:
            last_break, last_break_bsp, last_break_zs_begin_time, last_break_zs_end_time, last_break_zs_high, trade_df =\
            train_buy_model(code=code, begin_time = "2021-01-01", end_time = "2024-02-24")
        except Exception as e:
            print(e)
            continue
        res_dict['last_break'].append(last_break)
        res_dict['last_break_bsp'].append(last_break_bsp)
        res_dict['last_break_zs_begin_time'].append(last_break_zs_begin_time)
        res_dict['last_break_zs_end_time'].append(last_break_zs_end_time)
        res_dict['last_break_zs_high'].append(last_break_zs_high)

        res_dict['code'].append(code)
        all_code_df = pd.concat([all_code_df, trade_df], axis=0)
    res_df = pd.DataFrame(res_dict)
    print(res_df)
    print(all_code_df)
