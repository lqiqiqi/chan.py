#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 15:50
# @Author  : rockieluo
# @File    : buy_data_generation.py


import json
import math
from typing import Dict, TypedDict
import sys

sys.path.append('/root/chan.py')

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


label_dict = {'MRTYmain': 6, 'MYMmain': 50, 'MNQmain': 5}


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
            "x_range": 1000,
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


def find_max_klu(cur_lv_chan):
    max_klu_high = max([klu.high for ckl in cur_lv_chan[-14:] for klu in ckl][-14:-2])
    for ckl in cur_lv_chan[-14:]:
        for klu in ckl:
            if klu.high == max_klu_high:
                return klu


def find_low_macd_klu(cur_lv_chan, last_bsp):
    """最近的笔的低点，且macd最低，用于判断是否背离（当前买点，价格更低，macd却没有更低）"""
    last_bsp_macd = last_bsp.klu.macd.macd
    last_bsp_low = last_bsp.klu.low
    bi_begin_macd_dict = {bi.get_begin_klu().macd.macd: bi.get_begin_klu().low for bi in cur_lv_chan.bi_list[-3:]
                          if bi.idx >= last_bsp.bi.idx - 2}
    lowest_macd_low = last_bsp_low
    for klu_macd, klu_low in bi_begin_macd_dict.items():
        if klu_macd < last_bsp_macd:
            lowest_macd_low = klu_low
    return lowest_macd_low


def find_low_kdj_klu(cur_lv_chan, last_bsp):
    """最近的笔的低点，且kdj最低，用于判断是否背离"""
    last_bsp_kdj = last_bsp.klu.kdj.k
    last_bsp_low = last_bsp.klu.low
    bi_begin_kdj_dict = {bi.get_begin_klu().kdj.k: bi.get_begin_klu().low for bi in cur_lv_chan.bi_list[-3:]
                         if bi.idx >= last_bsp.bi.idx - 2}
    lowest_kdj_low = last_bsp_low
    for klu_kdj, klu_low in bi_begin_kdj_dict.items():
        if klu_kdj < last_bsp_kdj:
            lowest_kdj_low = klu_low
            # print(last_bsp.klu.time.to_str(), last_bsp_kdj, last_bsp_low, klu_kdj, lowest_kdj_low)
    return lowest_kdj_low


def candle_indicator(klu_list):
    data = {
        "open": [p.open for p in klu_list],
        "high": [p.high for p in klu_list],
        'low': [p.low for p in klu_list],
        'close': [p.close for p in klu_list],
        'time': [p.time.to_str() for p in klu_list]
    }
    df = pd.DataFrame(data)
    candles_df = df.sort_values(by='time', ascending=True)
    candles_df = candlestick.inverted_hammer(candles_df, target='倒锤子线（底部看涨，需要验证次日收市高于锤子线实体）')
    candles_df = candlestick.doji_star(candles_df, target='十字星')
    candles_df = candlestick.bearish_harami(candles_df, target='看跌孕线')
    candles_df = candlestick.bullish_harami(candles_df, target='看涨孕线')
    candles_df = candlestick.dark_cloud_cover(candles_df, target='乌云盖顶')
    candles_df = candlestick.doji(candles_df, target='十字线')
    candles_df = candlestick.dragonfly_doji(candles_df, target='蜻蜓十字线（看涨）')
    candles_df = candlestick.hanging_man(candles_df, target='上吊线（顶部看跌，需要验证次日收市低于上吊线实体）')
    candles_df = candlestick.gravestone_doji(candles_df, target='墓碑十字线（看跌）')
    candles_df = candlestick.bearish_engulfing(candles_df, target='看跌吞没')
    candles_df = candlestick.bullish_engulfing(candles_df, target='看涨吞没')
    candles_df = candlestick.hammer(candles_df, target='锤子(底部看反转)')
    candles_df = candlestick.morning_star(candles_df, target='启明星')
    candles_df = candlestick.morning_star_doji(candles_df, target='启明十字')
    candles_df = candlestick.piercing_pattern(candles_df, target='穿刺')
    candles_df = candlestick.rain_drop(candles_df, target='雨滴（趋势受阻）')
    candles_df = candlestick.rain_drop_doji(candles_df, target='雨滴十字线（趋势受阻）')
    candles_df = candlestick.star(candles_df, target='星线（趋势受阻）')
    candles_df = candlestick.shooting_star(candles_df, target='流星线（看跌）')
    candles_df = candlestick.window(candles_df, target='跳空（形成支撑）')
    return candles_df


def buy_stragety_feature(last_klu, cur_lv_chan, bsp_list):
    last_bsp = bsp_list[-1]
    candles_df = candle_indicator([klu for ckl in cur_lv_chan[-15:] for klu in ckl][-6:])
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
        "high_close_rate": (last_klu.high - last_klu.close)/last_klu.close,
        "low_close_rate": (last_klu.low - last_klu.close) / last_klu.close,
        "macd": last_klu.macd.macd,
        "distance_boll_up": (last_klu.close - last_klu.boll.UP)/last_klu.close,
        "distance_boll_down": (last_klu.close - last_klu.boll.DOWN)/last_klu.close,
        "kd": last_klu.kdj.k - last_klu.kdj.d,
        "k": last_klu.kdj.k,
        "voc": last_klu.voc,
        "ma10": (last_klu.trend[TREND_TYPE.MEAN][5] - last_klu.trend[TREND_TYPE.MEAN][10])/last_klu.trend[TREND_TYPE.MEAN][5],
        "ma20": (last_klu.trend[TREND_TYPE.MEAN][5] - last_klu.trend[TREND_TYPE.MEAN][20])/last_klu.trend[TREND_TYPE.MEAN][5],
        "recent_bar_avg": np.mean([(klu.close - klu.open) / klu.open for ckl in cur_lv_chan[-3:] for klu in ckl][-3:]), # 近期bar的长度，阴线为负
        # "is_buy": 1 if last_bsp.is_buy else 0,
        "voma10": (last_klu.trade_info.metric['volume'] - last_klu.voma.voma10)/last_klu.voma.voma10,
        "voma_diff": last_klu.voma.voma_diff/last_klu.voma.voma10,
        "retrace_rate": (cur_lv_chan.bi_list[-1].get_begin_klu().low - last_klu.close) / cur_lv_chan.bi_list[
            -1].get_begin_klu().low if len(cur_lv_chan.bi_list) > 0 else 0,
        "seg_retrace_rate": (cur_lv_chan.seg_list[-1].get_begin_klu().close - last_klu.close) / cur_lv_chan.seg_list[
            -1].get_begin_klu().close if len(cur_lv_chan.seg_list) > 0 else 0,
        "recent_bsp": sum([1 if not bsp.is_buy else 0 for bsp in bsp_list[-5:]]),
        "recent_high_macd": last_klu.macd.macd - find_max_klu(cur_lv_chan).macd.macd,
        "recent_low_macd": last_bsp.klu.low - find_low_macd_klu(cur_lv_chan, last_bsp),  # 背离
        "recent_low_kdj": last_bsp.klu.low - find_low_kdj_klu(cur_lv_chan, last_bsp),  # 背离
        "recent_high_divergence": (last_klu.macd.macd - find_max_klu(cur_lv_chan).macd.macd) /
                                  (last_klu.high - find_max_klu(cur_lv_chan).high + 0.01),
        "inverted_hammer": candles_df['倒锤子线（底部看涨，需要验证次日收市高于锤子线实体）'].sum(),
        "doji_star": candles_df['十字星'].sum(),
        "bearish_harami": candles_df['看跌孕线'].sum(),
        "bullish_harami": candles_df['看涨孕线'].sum(),
        "dark_cloud_cover": candles_df['乌云盖顶'].sum(),
        "doji": candles_df['十字线'].sum(),
        "dragonfly_doji": candles_df['蜻蜓十字线（看涨）'].sum(),
        "hanging_man": candles_df['上吊线（顶部看跌，需要验证次日收市低于上吊线实体）'].sum(),
        "gravestone_doji": candles_df['墓碑十字线（看跌）'].sum(),
        "bearish_engulfing": candles_df['看跌吞没'].sum(),
        "bullish_engulfing": candles_df['看涨吞没'].sum(),
        "hammer": candles_df['锤子(底部看反转)'].sum(),
        "morning_star": candles_df['启明星'].sum(),
        "morning_star_doji": candles_df['启明十字'].sum(),
        "piercing_pattern": candles_df['穿刺'].sum(),
        "rain_drop": candles_df['雨滴（趋势受阻）'].sum(),
        "rain_drop_doji": candles_df['雨滴十字线（趋势受阻）'].sum(),
        "star": candles_df['星线（趋势受阻）'].sum(),
        "shooting_star": candles_df['流星线（看跌）'].sum(),
        "window": candles_df['跳空（形成支撑）'].sum(),
        'bsp_type_1': 1 if BSP_TYPE.T1P in last_bsp.type or BSP_TYPE.T1 in last_bsp.type else 0,
        'bsp_type_2': 1 if BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type else 0,
        'bsp_type_3': 1 if BSP_TYPE.T3A in last_bsp.type or BSP_TYPE.T3B in last_bsp.type or
                             BSP_TYPE.T3S in last_bsp.type else 0,
        # "bsp_macd_diff": (last_bsp.klu.macd.macd - last_klu.macd.macd)/(last_bsp.klu.close - last_klu.close + 0.01)
    }


def train_buy_model(code, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    data_src = DATA_SRC.CSV
    # data_src = DATA_SRC.TigerMock
    lv_list = [KL_TYPE.K_5M]

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

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    first_indx = 0
    for chan_snapshot in chan.step_load():
        if first_indx < 200:
            first_indx += 1
            continue
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
        # 这里不能写or cur_lv_chan[-3].idx == last_bsp.klu.klc.idx，因为bsp_dict是按照bsp来存的，一个bsp特征是唯一的

        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-3].idx == last_bsp.klu.klc.idx and last_bsp.is_buy:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                # "feature": CFeatures({}),
                "is_buy": last_bsp.is_buy,
                "open_time": last_bsp.klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(buy_stragety_feature(last_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
            # print(last_bsp.klu.time, last_klu.time.to_str(), last_bsp.is_buy,
            #       [(fid, f) for fid, f in bsp_dict[last_bsp.klu.idx]['feature'].items()])
            print(last_bsp.klu.time, last_klu.time.to_str())

    # 生成libsvm样本特征
    # bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp() if bsp.is_buy]
    def find_closest_larger_number(number, num_list):
        larger_numbers = [x for x in num_list if x > number]
        if not larger_numbers:
            return None
        return min(larger_numbers)

    # bsp_academy = []
    # bsp_up_dict = {}
    # for bi in chan[0].bi_list:
    #     if bi.is_up():
    #         bsp_academy.append(bi.get_begin_klu().idx)
    #         bsp_up_dict[bi.get_begin_klu().idx] = bi.get_end_klu().close - bi.get_begin_klu().close

    bi_end_dict = {bi.get_end_klu().idx: bi.get_end_klu().close for bi in chan[0].bi_list}
    bi_end_list = [bi.get_end_klu().idx for bi in chan[0].bi_list]
    ck_idx_dict = {}
    for ck in chan[KL_TYPE.K_5M]:
        for klu in ck:
            ck_idx_dict[klu.idx] = klu.close
    bi_end_time = {bi.get_end_klu().idx: bi.get_end_klu().time.to_str() for bi in chan[0].bi_list}
    ck_time_dict = {}
    for ck in chan[KL_TYPE.K_5M]:
        for klu in ck:
            ck_time_dict[klu.idx] = klu.time.to_str()

    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fid = open(f"buy_feature_{code}_20240128_2.libsvm", "w")
    for bsp_klu_idx, feature_info in bsp_dict.items():
        # label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label，如果在bsp_academy中即为正确（后视镜看它是否正确）
        end_bi_indx = find_closest_larger_number(bsp_klu_idx, bi_end_list)
        # 该买点到该笔结束，至少要有多少的利润，才算正样本。可以参考np.percentile([bi.get_end_klu().close - bi.get_begin_klu().low for bi in chan[0].bi_list if bi.get_end_klu().close - bi.get_begin_klu().low > 0], 50)
        # MYM
        if (not end_bi_indx) or bi_end_dict[end_bi_indx] - ck_idx_dict[bsp_klu_idx] < label_dict[code]:
            label = 0
        else:
            # label = bi_end_dict[end_bi_indx] - ck_idx_dict[bsp_klu_idx]
            label = 1
        # label = 1 / (1 + math.exp(-label))
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

    with open(f"buy_feature_{code}_20240128_2.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    X, y = load_svmlight_file(f"buy_feature_{code}_20240128_2.libsvm")    # load sample
    print(np.unique(y, return_counts=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define parameters
    if code == 'MNQmain':
        param = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 3.6}
    elif code == 'MRTYmain':
        param = {'max_depth': 30, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 3.3, 'subsample': 0.9}
    else:
        param = {'max_depth': 30, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'scale_pos_weight': 2.3, 'subsample': 0.9}
        # param = {'max_depth': 20, 'eta': 0.05, 'objective': 'reg:logistic', 'eval_metric': 'rmse'}

    # Train model
    evals_result = {}
    bst = xgb.train(
        param,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtest, "test")],
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=True,
    )
    bst.save_model(f"buy_model_{code}_20240128_2.json")

    # Evaluate model
    preds = bst.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    print(f"test AUC: {auc}")

    # 全量训练
    dtotal = xgb.DMatrix(f"buy_feature_{code}_20240128_2.libsvm?format=libsvm")  # load sample

    evals_result = {}
    bst_total = xgb.train(
        param,
        dtrain=dtotal,
        num_boost_round=10,
        evals=[(dtotal, "train")],
        evals_result=evals_result,
        verbose_eval=True,
    )
    bst_total.save_model(f"buy_model_{code}_20240128_2.json")

    plot(chan, plot_marker)


if __name__ == '__main__':
    # 不要包含未来！
    train_buy_model(code='MRTYmain', begin_time="2019-05-20 00:00:00", end_time="2023-12-01 00:00:00")
