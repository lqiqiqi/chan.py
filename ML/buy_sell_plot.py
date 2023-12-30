#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 14:49
# @Author  : rockieluo
# @File    : buy_sell_plot.py
import json
from typing import Dict

import xgboost as xgb

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE
from Plot.PlotDriver import CPlotDriver
from buy_data_generation import buy_stragety_feature
from config import Config
from sell_data_generation import sell_stragety_feature


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 600,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("backtest_prob.png")


def predict_bsp(model: xgb.Booster, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)
    return model.predict(dtest)


if __name__ == "__main__":
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    code = "QQQ"
    begin_time = "2021-01-01"
    end_time = "2023-12-20"
    data_src = DATA_SRC.YFINANCE
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

    sell_model = xgb.Booster()
    sell_model.load_model("sell_model.json")
    sell_meta = json.load(open("sell_feature.meta", "r"))

    buy_model = xgb.Booster()
    buy_model.load_model("buy_model.json")
    buy_meta = json.load(open("buy_feature.meta", "r"))

    is_hold = False
    treated_bsp_idx = set()
    prob_dict = {'bsp_time': [], 'last_klu_time': [], 'prob': [], 'bsp_type': []}
    plot_marker = {}
    used_bsp_list = []
    buy_cnt = 0
    max_price = 0
    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        cur_lv_chan = chan_snapshot[0]

        # if is_hold and not last_bsp.is_buy:
        last_bsp.features.add_feat(sell_stragety_feature(last_klu, cur_lv_chan))  # 开仓K线特征
        sell_pred_prob = predict_bsp(sell_model, last_bsp, sell_meta)[0]

        prob_dict['bsp_time'].append(last_bsp.klu.time)
        prob_dict['last_klu_time'].append(last_klu.time)
        prob_dict['prob'].append(sell_pred_prob)
        prob_dict['bsp_type'].append(last_bsp.is_buy)
        treated_bsp_idx.add(last_bsp.klu.idx)

        if sell_pred_prob > 0.8:
            plot_marker[last_klu.time.to_str()] = (
                sell_pred_prob, "up")
            sell_price = cur_lv_chan[-1][-1].close
            # print(
            #     f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, '
            #     f'profit rate = {(sell_price - last_buy_price) / last_buy_price * 100:.2f}%')
            is_hold = False
        print(last_klu.time.to_str(), 'sell prob is ', sell_pred_prob)

        # if is_hold:
        #     max_price = max(cur_lv_chan[-1][-1].close, max_price)
        #     print(cur_lv_chan[-1][-1].time, 'hold, max price is ', max_price)
        #     if (max_price - cur_lv_chan[-1][-1].close)/max_price > 0.02:
        #         sell_price = cur_lv_chan[-1][-1].close
        #         print(
        #             f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, '
        #             f'profit rate = {(sell_price - last_buy_price) / last_buy_price * 100:.2f}%')
        #         is_hold = False

        # if not is_hold:
        last_bsp.features.add_feat(buy_stragety_feature(last_klu, cur_lv_chan, last_bsp))  # 开仓K线特征
        # print(last_bsp.klu.time, predict_bsp(buy_model, last_bsp, buy_meta))

        buy_pred_prob = predict_bsp(buy_model, last_bsp, buy_meta)[0]

        prob_dict['bsp_time'].append(last_bsp.klu.time)
        prob_dict['last_klu_time'].append(last_klu.time)
        prob_dict['prob'].append(buy_pred_prob)
        prob_dict['bsp_type'].append(last_bsp.is_buy)
        treated_bsp_idx.add(last_bsp.klu.idx)
        # buy_cnt += 1
        if buy_pred_prob > 0.8:
            plot_marker[last_klu.time.to_str()] = (
                buy_pred_prob, "down", 'red')
            # last_buy_price = cur_lv_chan[-1][-1].close  # 开仓价格为最后一根K线close
            # print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            # used_bsp_list.append(last_bsp.klu.time)
            is_hold = True
            # max_price = last_buy_price
            # buy_cnt = 0
        print(last_klu.time.to_str(), 'buy prob is ', buy_pred_prob)

    plot(chan, plot_marker)