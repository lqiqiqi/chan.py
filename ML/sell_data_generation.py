#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 15:50
# @Author  : rockieluo
# @File    : sell_data_generation.py


import json
from typing import Dict, TypedDict

import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from Test.config import Config


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


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
    plot_driver.save2img("sell_label.png")


def find_min_klu(cur_lv_chan):
    min_klu_low = min([klu.low for ckl in cur_lv_chan[-5:] for klu in ckl][-14:-2])
    for ckl in cur_lv_chan[-5:]:
        for klu in ckl:
            if klu.low == min_klu_low:
                return klu

def sell_stragety_feature(last_klu, cur_lv_chan):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
        "high_close_rate": (last_klu.high - last_klu.close)/last_klu.close,
        "low_close_rate": (last_klu.low - last_klu.close) / last_klu.close,
        "macd": last_klu.macd.macd,
        "distance_boll_up": last_klu.close - last_klu.boll.UP,
        "distance_boll_down": last_klu.close - last_klu.boll.DOWN,
        "kd": last_klu.kdj.k - last_klu.kdj.d,
        "k": last_klu.kdj.k,
        "voc": last_klu.voc,
        # "ma60": last_klu.close - last_klu.trend[TREND_TYPE.MEAN][60],
        "voma10": last_klu.trade_info.metric['volume'] - last_klu.voma.voma10,
        "voma_diff": last_klu.voma.voma_diff,
        "retrace_rate": (cur_lv_chan.bi_list[-1].get_begin_klu().low - last_klu.close)/cur_lv_chan.bi_list[-1].get_begin_klu().high if len(cur_lv_chan.bi_list)>0 else 0,
        "recent_high_macd": last_klu.macd.macd - find_min_klu(cur_lv_chan).macd.macd,
        "recent_high_divergence": (last_klu.macd.macd - find_min_klu(cur_lv_chan).macd.macd) /
                                  (last_klu.low - find_min_klu(cur_lv_chan).low + 0.01),
    }


def train_sell_model(code, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    data_src = DATA_SRC.YFINANCE
    lv_list = [KL_TYPE.K_DAY]

    config_object = Config()
    chan_config = config_object.read_chan_config_trigger_step
    config = CChanConfig(chan_config)
    # config = CChanConfig({
    #     "triger_step": True,  # 打开开关！
    #     "mean_metrics": [60],
    #     "cal_kdj": True,
    #     "cal_rsi": True,
    #     "cal_vol_change": True
    # })


    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        # if last_bsp.is_buy:
            # print(last_bsp.klu.time)

        cur_lv_chan = chan_snapshot[0]
        # if last_bsp.klu.idx not in bsp_dict and \
        #         (cur_lv_chan[-2].idx == last_bsp.klu.klc.idx or cur_lv_chan[-3].idx == last_bsp.klu.klc.idx) and \
        #         last_bsp.is_buy:
        if (
                cur_lv_chan[-2].idx == last_bsp.klu.klc.idx
                # cur_lv_chan[-3].idx == last_bsp.klu.klc.idx
        ):
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                # "feature": last_bsp.features,
                "feature": CFeatures({}),
                "is_buy": last_bsp.is_buy,
                "open_time": last_bsp.klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(sell_stragety_feature(last_klu, cur_lv_chan))  # 开仓K线特征
            print(last_klu.time, last_bsp.is_buy)

    # 生成libsvm样本特征
    # bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp() if not bsp.is_buy]
    bsp_academy = []
    for bi in chan[0].bi_list:
        if not bi.is_up():
            bsp_academy.append(bi.get_begin_klu().idx)

    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fid = open(f"sell_feature_{code}.libsvm", "w")
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label，如果在bsp_academy中即为正确（后视镜看它是否正确）
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():
            if feature_name not in feature_meta:
                feature_meta[feature_name] = cur_feature_idx
                cur_feature_idx += 1
            features.append((feature_meta[feature_name], value))
        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        fid.write(f"{label} {feature_str}\n")
        plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
    fid.close()

    with open(f"sell_feature_{code}.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    dtrain = xgb.DMatrix(f"sell_feature_{code}.libsvm?format=libsvm")  # load sample
    print(np.unique(dtrain.get_label(), return_counts=True))
    param = {'max_depth': 4, 'eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'scale_pos_weight': 3}
    evals_result = {}
    bst = xgb.train(
        param,
        dtrain=dtrain,
        num_boost_round=15,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=True,
    )
    bst.save_model(f"sell_model_{code}.json")

    # Evaluate model
    preds = bst.predict(dtrain)
    auc = roc_auc_score(dtrain.get_label(), preds)
    print(f"test AUC: {auc}")

    # load model
    model = xgb.Booster()
    model.load_model(f"sell_model_{code}.json")
    # predict
    print(model.predict(dtrain))

    plot(chan, plot_marker)


if __name__ == '__main__':
    train_sell_model(code='IWM', begin_time="2001-01-01", end_time="2023-01-01")
