# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 15:50
# @Author  : rockieluo
# @File    : buy_data_generation.py


import json
import math
import dill
from typing import Dict, TypedDict
import sys
from joblib import load
from DataAPI.csvAPI import CSV_API

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
print(__file__)  # 输出完整路径


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    bsp_time: CTime


def save_obj(obj, file_name):
    with open(f'{file_name}.pkl', 'wb') as f:
        dill.dump(obj, f)
    # with open("example_pickle.pkl", "rb") as f:
    #     loaded_dict = pickle.load(f)


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
    """近24个klu的最大值，找到该klu，24是两个小时"""
    klu_list = [klu.high for ckl in cur_lv_chan[-24:] for klu in ckl][-24:-2]
    max_klu_high = max(klu_list)
    for ckl in cur_lv_chan[-24:]:
        for klu in ckl:
            if klu.high == max_klu_high:
                return klu


def find_recent_lowest_macd_for_bsp(cur_lv_chan, last_bsp):
    klu_list = [klu.macd.macd for ckl in cur_lv_chan[-24:] for klu in ckl][-24:-2]
    if not len(klu_list):
        return 0
    min_macd = min(klu_list)
    for ckl in cur_lv_chan[-24:]:
        for klu in ckl:
            if klu.macd.macd == min_macd:
                # print(last_bsp.klu.time.to_str(), last_bsp.klu.low, last_bsp.klu.macd.macd, klu.time.to_str(), min_macd, klu.low)
                if last_bsp.klu.low < klu.low and last_bsp.klu.macd.macd > min_macd:
                    return last_bsp.klu.macd.macd - min_macd
                else:
                    return 0


def find_recent_lowest_macd_for_klu(cur_lv_chan):
    last_bi = cur_lv_chan.bi_list[-1]
    bi_begin_klu = last_bi.get_begin_klu()
    klu_list = [klu.macd.macd for ckl in cur_lv_chan[-24:] for klu in ckl][-24:-2]
    if not len(klu_list):
        return 0
    min_macd = min(klu_list)
    for ckl in cur_lv_chan[-24:]:
        for klu in ckl:
            if klu.macd.macd == min_macd:
                # print(last_bsp.klu.time.to_str(), last_bsp.klu.low, last_bsp.klu.macd.macd, klu.time.to_str(), min_macd, klu.low)
                if bi_begin_klu.low < klu.low and bi_begin_klu.macd.macd > min_macd:
                    return bi_begin_klu.macd.macd - min_macd
                else:
                    return 0


def find_recent_lowest_macd_before_bsp(cur_lv_chan, last_bsp):
    klu_list = [klu.macd.macd for ckl in cur_lv_chan[-24:] for klu in ckl if klu.idx < last_bsp.klu.idx]
    if not len(klu_list):
        return 0
    min_macd = min(klu_list)
    for ckl in cur_lv_chan[-27:]:
        for klu in ckl:
            if klu.macd.macd == min_macd:
                if last_bsp.klu.low < klu.low and last_bsp.klu.macd.macd > min_macd:
                    # print(last_bsp.klu.time.to_str(), last_bsp.klu.low, last_bsp.klu.macd.macd, klu.time.to_str(),
                    #       min_macd,
                    #       klu.low)
                    return last_bsp.klu.macd.macd - min_macd
                else:
                    return 0


def find_low_macd_klu_for_bsp(cur_lv_chan, last_bsp):
    """最近的笔的低点，且macd最低，用于判断是否背离（macd比当前bsp的macd更低，但是low却比当前bsp要高，说明macd这里背离了）"""
    last_bsp_macd = last_bsp.klu.macd.macd
    last_bsp_low = last_bsp.klu.low
    bi_begin_macd_dict = {bi.get_begin_klu().macd.macd: bi.get_begin_klu().low for bi in cur_lv_chan.bi_list[-3:]
                          if bi.idx >= last_bsp.bi.idx - 2}
    lowest_macd_low = last_bsp_low
    for klu_macd, klu_low in bi_begin_macd_dict.items():
        if klu_macd < last_bsp_macd:
            lowest_macd_low = klu_low
    return lowest_macd_low


def find_low_macd_klu_for_klu(cur_lv_chan):
    """最近的笔的低点，且macd最低，用于判断是否背离（macd比当前bsp的macd更低，但是low却比当前bsp要高，说明macd这里背离了）"""
    last_bi = cur_lv_chan.bi_list[-1]
    last_klu_macd = last_bi.get_begin_klu().macd.macd
    last_klu_low = last_bi.get_begin_klu().low
    bi_begin_macd_dict = {bi.get_begin_klu().macd.macd: bi.get_begin_klu().low for bi in cur_lv_chan.bi_list[-3:]
                          if bi.idx >= last_bi.idx - 2}
    lowest_macd_low = last_klu_low
    for klu_macd, klu_low in bi_begin_macd_dict.items():
        if klu_macd < last_klu_macd:
            lowest_macd_low = klu_low
    return lowest_macd_low


def find_low_kdj_klu_for_bsp(cur_lv_chan, last_bsp):
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


def find_low_kdj_klu_for_klu(cur_lv_chan):
    """最近的笔的低点，且kdj最低，用于判断是否背离"""
    last_bi = cur_lv_chan.bi_list[-1]
    last_klu_kdj = last_bi.get_begin_klu().kdj.k
    last_klu_low = last_bi.get_begin_klu().low
    bi_begin_kdj_dict = {bi.get_begin_klu().kdj.k: bi.get_begin_klu().low for bi in cur_lv_chan.bi_list[-3:]
                         if bi.idx >= last_bi.idx - 2}
    lowest_kdj_low = last_klu_low
    for klu_kdj, klu_low in bi_begin_kdj_dict.items():
        if klu_kdj < last_klu_kdj:
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


def cal_ma_cross_times(cur_lv_chan, indx):
    short_ma_list = [klu.trend[TREND_TYPE.MEAN][5] for ckl in cur_lv_chan[-100:] for klu in ckl][-indx:]
    long_ma_list = [klu.trend[TREND_TYPE.MEAN][10] for ckl in cur_lv_chan[-100:] for klu in ckl][-indx:]
    if len(short_ma_list) != len(long_ma_list):
        return np.nan
    diff_list = [short_ma - long_ma for short_ma, long_ma in zip(short_ma_list, long_ma_list)]
    cross_count = 0
    for i in range(1, len(diff_list)):
        if diff_list[i] * diff_list[i - 1] < 0:
            cross_count += 1
    return cross_count


def cal_ma_slope(cur_lv_chan, ma_lv, indx):
    ma_list = [klu.trend[TREND_TYPE.MEAN][ma_lv] for ckl in cur_lv_chan[-200:] for klu in ckl][-indx:]
    # 使用numpy的polyfit函数来拟合数据
    try:
        slope, _ = np.polyfit(range(len(ma_list)), lst, 1)
    except:
        slope = 0
    return slope


def cal_bsp_vol_percentile(last_bsp, cur_lv_chan, indx):
    last_bsp_vol = last_bsp.klu.trade_info.metric['volume']
    vol_list = [klu.trade_info.metric['volume'] for ckl in cur_lv_chan[-indx*10:] for klu in ckl if
                klu.idx <= last_bsp.klu.idx][-indx:]
    vol_list.sort()
    try:
        rank = vol_list.index(last_bsp_vol) + 1
    except:
        return np.nan
    quantile = rank / len(vol_list)
    return quantile


def cal_bsp_voma3_percentile(last_bsp, cur_lv_chan, indx):
    last_bsp_vol = last_bsp.klu.voma.voma3
    vol_list = [klu.voma.voma3 for ckl in cur_lv_chan[-indx*10:] for klu in ckl if klu.idx <= last_bsp.klu.idx][-indx:]
    vol_list.sort()
    try:
        rank = vol_list.index(last_bsp_vol) + 1
    except:
        return np.nan
    quantile = rank / len(vol_list)
    return quantile


def cal_klu_voma3_percentile(last_klu, cur_lv_chan, indx):
    last_vol = last_klu.voma.voma3
    vol_list = [klu.voma.voma3 for ckl in cur_lv_chan[-indx*10:] for klu in ckl if klu.idx <= last_klu.idx][-indx:]
    vol_list.sort()
    try:
        rank = vol_list.index(last_vol) + 1
    except:
        return np.nan
    quantile = rank / len(vol_list)
    return quantile


def choppiness_index(cur_lv_chan, n=14):
    high = np.array([klu.high for ckl in cur_lv_chan[-100:] for klu in ckl])
    low = np.array([klu.low for ckl in cur_lv_chan[-100:] for klu in ckl])
    close = np.array([klu.close for ckl in cur_lv_chan[-100:] for klu in ckl])
    atr_sum = 0
    for i in range(n, 0, -1):
        tr = max(high[-i] - low[-i], abs(high[-i] - close[-(i + 1)]), abs(low[-i] - close[-(i + 1)]))
        atr_sum += tr

    atr = atr_sum / n
    ci = 100 * np.log10(sum(atr for _ in range(n)) / (high[-n:].max() - low[-n:].min())) / np.log10(n)
    return ci


def buy_stragety_feature(last_klu, cur_lv_chan, bsp_list):
    last_bsp = bsp_list[-1]
    candles_df = candle_indicator([klu for ckl in cur_lv_chan[-15:] for klu in ckl][-6:])
    return {
        "choppiness_index": choppiness_index(cur_lv_chan),
        "last_klu_last_bsp_distance": last_klu.idx - last_bsp.klu.idx,
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
        "high_close_rate": (last_klu.high - last_klu.close) / last_klu.close,
        "low_close_rate": (last_klu.low - last_klu.close) / last_klu.close,
        "bsp_low_close_rate": (last_bsp.klu.low - last_bsp.klu.close) / last_bsp.klu.close,
        "close_bsp_low_rate": (last_klu.close - last_bsp.klu.low) / last_bsp.klu.low,  # 距离bsp的低点已经涨了多少
        "close_bsp_close_rate": (last_klu.close - last_bsp.klu.close) / last_bsp.klu.close,
        "macd": last_klu.macd.macd,
        "distance_boll_up": (last_klu.close - last_klu.boll.UP) / last_klu.close,
        "distance_boll_down": (last_klu.close - last_klu.boll.DOWN) / last_klu.close,
        "kd": last_klu.kdj.k - last_klu.kdj.d,
        "k": last_klu.kdj.k,
        "voc": last_klu.voc,
        "ma10": (last_klu.trend[TREND_TYPE.MEAN][5] - last_klu.trend[TREND_TYPE.MEAN][10]) /
                last_klu.trend[TREND_TYPE.MEAN][5],
        "ma20": (last_klu.trend[TREND_TYPE.MEAN][5] - last_klu.trend[TREND_TYPE.MEAN][20]) /
                last_klu.trend[TREND_TYPE.MEAN][5],
        "long_head": 1 if last_klu.trend[TREND_TYPE.MEAN][5] > last_klu.trend[TREND_TYPE.MEAN][10] >
                          last_klu.trend[TREND_TYPE.MEAN][20] > last_klu.trend[TREND_TYPE.MEAN][60] else 0,
        "ma_cross_times_120min": cal_ma_cross_times(cur_lv_chan, 24),
        "ma_cross_times_60min": cal_ma_cross_times(cur_lv_chan, 12),
        "ma10_slope_60min": cal_ma_slope(cur_lv_chan, 10, 12),
        "ma20_slope_60min": cal_ma_slope(cur_lv_chan, 20, 12),
        "ma60_slope_60min": cal_ma_slope(cur_lv_chan, 60, 12),
        "ma10_slope_120min": cal_ma_slope(cur_lv_chan, 10, 24),
        "ma20_slope_120min": cal_ma_slope(cur_lv_chan, 20, 24),
        "ma60_slope_120min": cal_ma_slope(cur_lv_chan, 60, 24),
        "last_zs_high": last_klu.close - cur_lv_chan.zs_list[-1].high if len(cur_lv_chan.zs_list) > 0 else 0,
        "last_zs_low": last_klu.close - cur_lv_chan.zs_list[-1].low if len(cur_lv_chan.zs_list) > 0 else 0,
        "last_seg_zs_high": last_klu.close - cur_lv_chan.segzs_list[-1].high if len(cur_lv_chan.segzs_list) > 0 else 0,
        "last_seg_zs_low": last_klu.close - cur_lv_chan.segzs_list[-1].low if len(cur_lv_chan.segzs_list) > 0 else 0,
        "recent_bar_avg": np.mean([(klu.close - klu.open) / klu.open for ckl in cur_lv_chan[-3:] for klu in ckl][-3:]),
        # 近期bar的长度，阴线为负
        "voma10": (last_klu.trade_info.metric['volume'] - last_klu.voma.voma10) / last_klu.voma.voma10,
        "voma_diff": last_klu.voma.voma_diff / last_klu.voma.voma10,
        "bsp_vol_percentile_30min": cal_bsp_vol_percentile(last_bsp, cur_lv_chan, 6),
        "bsp_vol_percentile_60min": cal_bsp_vol_percentile(last_bsp, cur_lv_chan, 12),
        "bsp_vol_percentile_120min": cal_bsp_vol_percentile(last_bsp, cur_lv_chan, 24),
        "bsp_vol_percentile_1380min": cal_bsp_vol_percentile(last_bsp, cur_lv_chan, 276),  # 最近1天，交易23小时
        "last_klu_last_bsp_vol_diff": (last_klu.trade_info.metric['volume'] - last_bsp.klu.trade_info.metric[
            'volume']) / last_bsp.klu.trade_info.metric['volume'],
        "last_klu_last_bsp_voma3_diff": (last_klu.voma.voma3 - last_bsp.klu.voma.voma3) / last_bsp.klu.voma.voma3,
        "bsp_voma3_percentile_30min": cal_bsp_voma3_percentile(last_bsp, cur_lv_chan, 6),
        "bsp_voma3_percentile_60min": cal_bsp_voma3_percentile(last_bsp, cur_lv_chan, 12),
        "bsp_voma3_percentile_120min": cal_bsp_voma3_percentile(last_bsp, cur_lv_chan, 24),
        "bsp_voma3_percentile_1380min": cal_bsp_voma3_percentile(last_bsp, cur_lv_chan, 276),
        "klu_voma3_percentile_30min": cal_klu_voma3_percentile(last_klu, cur_lv_chan, 6),
        "klu_voma3_percentile_60min": cal_klu_voma3_percentile(last_klu, cur_lv_chan, 12),
        "klu_voma3_percentile_120min": cal_klu_voma3_percentile(last_klu, cur_lv_chan, 24),
        "klu_voma3_percentile_1380min": cal_klu_voma3_percentile(last_klu, cur_lv_chan, 276),
        "retrace_rate": (cur_lv_chan.bi_list[-1].get_begin_klu().low - last_klu.close) / cur_lv_chan.bi_list[
            -1].get_begin_klu().low if len(cur_lv_chan.bi_list) > 0 else 0,
        "seg_begin_rate": (cur_lv_chan.seg_list[-1].get_begin_klu().close - last_klu.close) / cur_lv_chan.seg_list[
            -1].get_begin_klu().close if len(cur_lv_chan.seg_list) > 0 else 0,  # 距离当前线段开头已经走了多远
        "bi_begin_rate": (cur_lv_chan.bi_list[-1].get_begin_klu().close - last_klu.close) / cur_lv_chan.bi_list[
            -1].get_begin_klu().close if len(cur_lv_chan.bi_list) > 0 else 0,  # 距离当前笔开头已经走了多远
        "last_seg_begin_rate": (cur_lv_chan.seg_list[-2].get_begin_klu().close - last_klu.close) /
                                 cur_lv_chan.seg_list[
                                     -2].get_begin_klu().close if len(cur_lv_chan.seg_list) > 1 else 0,
        # 距离上一个线段开头已经走了多远
        "last_bi_begin_rate": (cur_lv_chan.bi_list[-2].get_begin_klu().close - last_klu.close) /
                               cur_lv_chan.bi_list[
                                   -2].get_begin_klu().close if len(cur_lv_chan.bi_list) > 1 else 0,
        # 距离上一个笔开头已经走了多远
        "seg_up_down": 1 if cur_lv_chan.seg_list[-1].is_down() else 0,
        "recent_bsp": sum([1 if not bsp.is_buy else 0 for bsp in bsp_list[-3:]]),
        "recent_high_macd": last_klu.macd.macd - find_max_klu(cur_lv_chan).macd.macd,
        # 当前klu的macd和近14个klu中最高价的那个klu的macd，下降了多少
        "recent_low_macd_bsp": last_bsp.klu.low - find_low_macd_klu_for_bsp(cur_lv_chan, last_bsp),  # 背离
        "recent_low_kdj_bsp": last_bsp.klu.low - find_low_kdj_klu_for_bsp(cur_lv_chan, last_bsp),  # 背离
        "recent_low_macd_klu": last_klu.low - find_low_macd_klu_for_klu(cur_lv_chan),  # 背离
        "recent_low_kdj_klu": last_klu.low - find_low_kdj_klu_for_klu(cur_lv_chan),  # 背离
        "macd_dif_diff": last_bsp.klu.macd.DIF - last_klu.macd.DIF,
        "macd_cross": last_klu.macd.DIF - last_klu.macd.DEA,  # 金叉
        "recent_macd_diverge_for_bsp": find_recent_lowest_macd_for_bsp(cur_lv_chan, last_bsp),
        "recent_macd_diverge_for_klu": find_recent_lowest_macd_for_klu(cur_lv_chan),
        # bsp本身是否背离，最近的24个klu的macd最低，是不是就是bsp自己。如果不是，说明最近有更低的macd
        "recent_macd_diverge_before_bsp": find_recent_lowest_macd_before_bsp(cur_lv_chan, last_bsp),
        "recent_high_divergence": (last_klu.macd.macd - find_max_klu(cur_lv_chan).macd.macd) /
                                  (last_klu.high - find_max_klu(cur_lv_chan).high + 0.01),
        # 当前klu的macd和近14个klu中最高价的那个klu的macd，下降的比例
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

    data_src_5m = CSV_API(code, k_type=KL_TYPE.K_5M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    with open("bsp_dict_ma60_20240216_1.pkl", "rb") as f:
        bsp_dict = dill.load(f)
    # 跑策略，保存买卖点的特征
    first_indx = 0
    for last_5m_klu in data_src_5m.get_kl_data():
        chan.trigger_load({KL_TYPE.K_5M: [last_5m_klu]})
        if first_indx < 276:
            first_indx += 1
            continue
        chan_snapshot = chan
        last_last_5m_klu = [klu for ckl in chan_snapshot[0][-2:] for klu in ckl][-2]
        bsp_list = chan_snapshot.get_bsp()
        # seg_bsp_list = chan_snapshot.get_seg_bsp()
        if not bsp_list:
            continue
        last_bsp = [bsp for bsp in bsp_list if bsp.is_buy][-1]
        # if last_bsp.is_buy:
        # print(last_bsp.klu.time)

        cur_lv_chan = chan_snapshot[0]

        # if last_bsp.klu.idx not in bsp_dict and \
        #         (cur_lv_chan[-2].idx == last_bsp.klu.klc.idx or cur_lv_chan[-3].idx == last_bsp.klu.klc.idx) and \
        #         last_bsp.is_buy:
        # 这里不能写or cur_lv_chan[-3].idx == last_bsp.klu.klc.idx，因为bsp_dict是按照bsp来存的，一个bsp特征是唯一的

        if last_last_5m_klu.trend[TREND_TYPE.MEAN][5] < last_last_5m_klu.trend[TREND_TYPE.MEAN][60] and \
                last_5m_klu.trend[TREND_TYPE.MEAN][5] > last_5m_klu.trend[TREND_TYPE.MEAN][60] and \
                last_5m_klu.time.to_str() not in bsp_dict:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_5m_klu.time.to_str()] = {
                "feature": last_bsp.features,
                # "feature": CFeatures({}),
                "is_buy": last_bsp.is_buy,
                "bsp_time": last_bsp.klu.time,
                # "buy_time_klu_idx": last_5m_klu.idx,
            }
            bsp_dict[last_5m_klu.time.to_str()]['feature'].add_feat(
                buy_stragety_feature(last_5m_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
            # print(last_bsp.klu.time, last_klu.time.to_str(), last_bsp.is_buy,
            # [(fid, f) for fid, f in bsp_dict[last_5m_klu.time.to_str()]['feature'].items() if fid in ("last_klu_last_bsp_vol_diff","bsp_vol_percentile_30min","bsp_vol_percentile_60min","bsp_vol_percentile_120min","bsp_vol_percentile_1380min","last_klu_last_bsp_voma3_diff","bsp_voma3_percentile_30min","bsp_voma3_percentile_60min","bsp_voma3_percentile_120min","bsp_voma3_percentile_1380min")])
            print(last_bsp.klu.time, last_5m_klu.time.to_str())

    save_obj(bsp_dict, 'bsp_dict_ma60_20240216_1')

    label_dict = load(f'label_dict.joblib')

    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fid = open(f"buy_feature_{code}_ma60_20240216_1.libsvm", "w")
    for klu_time, feature_info in bsp_dict.items():
        if klu_time not in label_dict:
            continue
        label = 1 if klu_time in label_dict and label_dict[klu_time] > 0 else 0
        features = []  # List[(idx, value)]
        for feature_name, value in feature_info['feature'].items():

            if feature_name not in feature_meta:
                feature_meta[feature_name] = cur_feature_idx
                cur_feature_idx += 1
            features.append((feature_meta[feature_name], value))
        print(klu_time, label)
        features.sort(key=lambda x: x[0])
        feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        fid.write(f"{label} {feature_str}\n")
        plot_marker[klu_time] = (
            "√" if label else "×", "down" if feature_info["is_buy"] else "up")

    fid.close()

    with open(f"buy_feature_{code}_ma60_20240216_1.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    X, y = load_svmlight_file(f"buy_feature_{code}_ma60_20240216_1.libsvm")  # load sample
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
    bst.save_model(f"buy_model_{code}_ma60_20240216_1.json")

    # Evaluate model
    preds = bst.predict(dtest)
    auc = roc_auc_score(y_test, preds)
    print(f"test AUC: {auc}")

    # 全量训练
    dtotal = xgb.DMatrix(f"buy_feature_{code}_ma60_20240216_1.libsvm?format=libsvm")  # load sample

    evals_result = {}
    bst_total = xgb.train(
        param,
        dtrain=dtotal,
        num_boost_round=10,
        evals=[(dtotal, "train")],
        evals_result=evals_result,
        verbose_eval=True,
    )
    bst_total.save_model(f"buy_model_{code}_ma60_20240216_1.json")

    plot(chan, plot_marker)

    feature_importance = bst.get_score(importance_type='weight')
    sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1])


if __name__ == '__main__':
    # 不要包含未来！
    # begin_time = "2019-05-20 00:00:00", end_time = "2023-12-01 00:00:00"
    train_buy_model(code='MNQmain', begin_time="2023-10-25 00:00:00", end_time="2024-01-31 00:00:00")
