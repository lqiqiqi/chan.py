#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 20:27
# @Author  : rockieluo
# @File    : strategy_demo7.py


import pandas as pd
import joblib

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from Plot.PlotDriver import CPlotDriver
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import Config

if __name__ == "__main__":
    """
    模型预测
    """

    model = joblib.load('../Model/09868_nn.pkl')

    # code = "sz.000001"
    code = 'HK.00700'
    # code = 'HK.800000'
    # code = 'HK.09868'
    begin_time = "2023-01-01"
    end_time = "2023-10-31"
    # data_src = DATA_SRC.BAO_STOCK
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_DAY]

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

    is_hold = False
    last_buy_price = None
    used_bsp_list = []
    draw_dict = {'prediction': [], 'close_price': []}
    for chan_snapshot in chan.step_load():  # 每增加一根K线，返回当前静态精算结果
        data_dict = {
            # 'close_price': [],
            'macd': [],
            'direct': [],
            'bi_dir': [],
            'zs_high': [],
            'zs_low': [],
            'zs_begin_idx': [],
            'zs_end_idx': [],
            'is_buy': [],
            'one_bsp': [],
            'two_bsp': [],
            'two_s_bsp': [],
            'three_bsp': [],
            'last_bsp_idx': []
        }

        cur_lv_chan = chan_snapshot[0]

        # k线粒度
        cur_kline = cur_lv_chan[-1][-1]
        cur_date = cur_kline.time.toDateStr('-')
        if cur_date < begin_time:
            continue
        close_price = cur_kline.close
        klu_idx = cur_kline.idx
        macd = cur_kline.macd.macd

        # 合并后k线粒度
        cur_ckline = cur_lv_chan[-1]
        idx = cur_ckline.idx
        if FX_TYPE.BOTTOM == cur_ckline.fx:
            direct = 1
        elif FX_TYPE.TOP == cur_ckline.fx:
            direct = -1
        else:
            direct = 0

        if len(cur_lv_chan.bi_list) > 0:
            bi_dir = 1 if cur_lv_chan.bi_list[-1].is_up is True else 0
        else:
            bi_dir = 0

        # 中枢
        if len(cur_lv_chan.zs_list) > 0:
            zs_high = close_price - cur_lv_chan.zs_list[-1].high
            zs_low = close_price - cur_lv_chan.zs_list[-1].low
            zs_begin_idx = idx - cur_lv_chan.zs_list[-1].begin_bi.begin_klc.idx
            zs_end_idx = idx - cur_lv_chan.zs_list[-1].end_bi.end_klc.idx
        else:
            zs_high = 0
            zs_low = 0
            zs_begin_idx = 0
            zs_end_idx = 0

        # bsp
        bsp_list = chan_snapshot.get_bsp()  # 获取买卖点列表
        if len(bsp_list) > 0:
            last_bsp = bsp_list[-1]
            is_buy = 1 if last_bsp.is_buy is True else -1
            one_bsp = 1 if BSP_TYPE.T1P in last_bsp.type or BSP_TYPE.T1 in last_bsp.type else -1
            two_bsp = 1 if BSP_TYPE.T2 in last_bsp.type else -1
            two_s_bsp = 1 if BSP_TYPE.T2S in last_bsp.type else -1
            three_bsp = 1 if BSP_TYPE.T3B in last_bsp.type or BSP_TYPE.T3A in last_bsp.type else -1
            last_bsp_idx = klu_idx - last_bsp.klu.idx
        else:
            is_buy = 0
            one_bsp = 0
            two_bsp = 0
            two_s_bsp = 0
            three_bsp = 0
            last_bsp_idx = 0

        data_dict['macd'].append(macd)
        data_dict['direct'].append(direct)
        data_dict['bi_dir'].append(bi_dir)
        data_dict['zs_high'].append(zs_high)
        data_dict['zs_low'].append(zs_low)
        data_dict['zs_begin_idx'].append(zs_begin_idx)
        data_dict['zs_end_idx'].append(zs_end_idx)
        data_dict['is_buy'].append(is_buy)
        data_dict['one_bsp'].append(one_bsp)
        data_dict['two_bsp'].append(two_bsp)
        data_dict['two_s_bsp'].append(two_s_bsp)
        data_dict['three_bsp'].append(three_bsp)
        data_dict['last_bsp_idx'].append(last_bsp_idx)
        # data_dict['close_price'].append(close_price)

        input_df = pd.DataFrame(data_dict)
        prediction = model.predict_proba(input_df.values)[0][1]

        draw_dict['prediction'].append(prediction)
        draw_dict['close_price'].append(close_price)

    draw_y_df = pd.DataFrame(draw_dict)

    fig, ax1 = plt.subplots()
    ax1.plot(draw_y_df['prediction'], label='prediction', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(draw_y_df['close_price'], label='close_price', color='red')
    plt.legend()
    plt.show()

        # pred_pos_cnt = len(draw_y_df[draw_y_df['prediction'] > 0])
        # true_pos_cnt = len(draw_y_df[(draw_y_df['prediction'] > 0) & (draw_y_df['return_rate'] > 0)])

        # if prediction > 90:
        #     # and not is_hold
        #     plot_para.update({"marker": {'markers': {
        #         cur_kline.time.toDateStr('/'): ('BUY!!', 'up', 'red')
        #     }}})
        #     plot_driver = CPlotDriver(
        #         chan_snapshot,
        #         plot_config=plot_config,
        #         plot_para=plot_para,
        #     )
        #     plot_driver.figure.show()
        #     last_buy_price = cur_lv_chan[-1][-1].close  # 开仓价格为最后一根K线close
        #     print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            # used_bsp_list.append(last_bsp.klu.time)
            # is_hold = True

        # if prediction < -10 and is_hold:
        #     sell_price = cur_lv_chan[-1][-1].close
        #     plot_para.update({"marker": {'markers': {
        #         cur_kline.time.toDateStr('/'): ('SELL!!', 'up', 'red')
        #     }}})
        #     plot_driver = CPlotDriver(
        #         chan_snapshot,
        #         plot_config=plot_config,
        #         plot_para=plot_para,
        #     )
        #     plot_driver.figure.show()
        #     print(
        #         f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {(sell_price - last_buy_price) / last_buy_price * 100:.2f}%')
        #     is_hold = False
