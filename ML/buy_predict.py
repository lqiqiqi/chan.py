#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 17:34
# @Author  : rockieluo
# @File    : buy_predict.py


import json
from typing import Dict, TypedDict

import pandas as pd
import xgboost as xgb

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from ML.buy_data_generation import buy_stragety_feature
from Plot.PlotDriver import CPlotDriver
from Test.config import Config


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
    plot_driver.save2img("buy_label.png")



class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def predict_bsp(model: xgb.Booster, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)
    fea_list = []
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
            fea_list.append((feat_name, feat_value))
    print(fea_list)
    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)
    return model.predict(dtest)


if __name__ == "__main__":
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    code = "QQQ"
    begin_time = "2021-01-01"
    end_time = None
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

    model = xgb.Booster()
    model.load_model("buy_model.json")
    meta = json.load(open("buy_feature.meta", "r"))

    treated_bsp_idx = set()
    prob_dict = {'bsp_time': [], 'last_klu_time': [], 'prob': [], 'bsp_type': []}
    plot_marker = {}
    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        cur_lv_chan = chan_snapshot[0]
        if cur_lv_chan[-3].idx != last_bsp.klu.klc.idx or \
            last_bsp.klu.idx in treated_bsp_idx or not last_bsp.is_buy:
            # and cur_lv_chan[-2].idx != last_bsp.klu.klc.idx)
                # 已经判断过了，分型还没形成，不是买点
                continue
        # if not last_bsp.is_buy:
        #     continue

        last_bsp.features.add_feat(buy_stragety_feature(last_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
        # 买卖点打分，应该和demo5最后的predict结果完全一致才对
        print(last_bsp.klu.time, last_klu.time.to_str())

        pred_prob = predict_bsp(model, last_bsp, meta)[0]

        prob_dict['bsp_time'].append(last_bsp.klu.time)
        prob_dict['last_klu_time'].append(last_klu.time)
        prob_dict['prob'].append(pred_prob)
        prob_dict['bsp_type'].append(last_bsp.is_buy)
        treated_bsp_idx.add(last_bsp.klu.idx)
        if pred_prob > 0.65:
            plot_marker[last_klu.time.to_str()] = (
                pred_prob, "down" if last_bsp.is_buy else "up")


    df = pd.DataFrame(prob_dict)
    df = df.sort_values('last_klu_time')
    print(df)

    feature_importance = model.get_score(importance_type='weight')
    print(feature_importance)

    plot(chan, plot_marker)
