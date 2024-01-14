#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 11:53
# @Author  : rockieluo
# @File    : daily_monitor.py

# 先按金大第一定律判断现在应采取何种交易策略进场。
# 如果判断现在是趋势行情，那么按照本交易系统给出的安全仓位在合适的开仓点进场进行趋势交易，并按金大第三定律设置止损位。
# 如果判断现在是震荡行情，则同样使用安全的仓位采用震荡策略在合适的开仓点进行交易，并按照金大第三定律设置止损位。
# 出现反向波动后，根据金大第二定律判断行情是否出现趋势暂时中止而进入了震荡阶段，然后按照震荡阶段进行处理。

import sys
import time

import numpy as np
import pandas as pd

from candlestick import candlestick

sys.path.append('/root/chan.py')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE
from Test.config import Config
from get_image_api import send_msg


def ma_rank(last_klu):
    ma5 = last_klu.trend[TREND_TYPE.MEAN][5]
    ma10 = last_klu.trend[TREND_TYPE.MEAN][10]
    ma20 = last_klu.trend[TREND_TYPE.MEAN][20]
    ma60 = last_klu.trend[TREND_TYPE.MEAN][60]

    # 创建一个字典，将数值与它们的名称关联起来
    variables = {ma5: "ma5", ma10: "ma10", ma20: "ma20", ma60: "ma60"}
    # 对数值进行排序
    sorted_variables = sorted(variables.keys(), reverse=True)
    # 打印结果
    return_str = '均线大小排序: '
    var_list = []
    for value in sorted_variables:
        var_list.append(variables[value]+"(" + str(np.round(value, 2)) + ")")
    return_str += " > ".join(var_list)

    if ma5 < ma20:
        is_ma5_gt_ma20 = False
    else:
        is_ma5_gt_ma20 = True

    if ma5 < ma60:
        is_ma5_gt_ma60 = False
    else:
        is_ma5_gt_ma60 = True

    return return_str, is_ma5_gt_ma20, is_ma5_gt_ma60


def ma_cross(klu_list):
    res_list = []
    for klu in klu_list:
        _, is_ma5_gt_ma20, _ = ma_rank(klu)
        res_list.append(is_ma5_gt_ma20)

    # 使用all()函数判断列表中的所有元素是否一致
    result = all(x == res_list[0] for x in res_list)

    if result:
        return_str = '近15个交易日没有发生ma5穿越ma20'
    else:
        return_str = '近15个交易日发生了ma5穿越ma20'
    return return_str


def find_true_columns(df):
    results = []
    df = df.dropna()
    # 从最后一行开始倒推
    for index in df.index:
        # 获取布尔值列
        bool_columns = df.drop(["time", 'open', 'high', 'low', 'close'], axis=1)
        # 查找当前行中为True的列
        true_columns = bool_columns.columns[bool_columns.loc[index]].tolist()
        # 如果找到了True的列
        if true_columns:
            # 获取"time"字段值
            time_value = df.loc[index, "time"]
            # 将结果添加到结果列表中
            results.append((time_value, true_columns))
    res_str = ''
    for tup in results:
        # 使用str()函数将tuple转换为string
        str_tup = str(tup)
        # 打印转换后的string
        res_str += ' '
        res_str += str_tup
    return res_str


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
    res_str = find_true_columns(candles_df)
    return res_str



def daily_indicator(last_klu, cur_lv_chan):
    return_str = f' 最新数据日期 {last_klu.time.to_str()} \n'
    return_str += '1. 判断当前应该是趋势还是震荡 \n'
    return_str += 'a. 均线判断方法：日线均线是否整排列'
    ma_rank_str, _, _ = ma_rank(last_klu)
    return_str += ma_rank_str

    klu_list = [klu for ckl in cur_lv_chan[-15:] for klu in ckl]
    ma_cross_str = ma_cross(klu_list)
    return_str += ', ' + ma_cross_str
    candle_str = candle_indicator(klu_list)
    return_str += '\n'
    return_str += candle_str
    return return_str


def daily_cal(code, begin_time, end_time=None):
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

    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        last_klu = chan_snapshot[0][-1][-1]
        cur_lv_chan = chan_snapshot[0]

    return_str = code
    return_str += daily_indicator(last_klu, cur_lv_chan)
    return return_str


if __name__ == '__main__':
    return_str = ''
    for code in ['QQQ', 'IWM']:
        try:
            return_str += daily_cal(code=code, begin_time="2023-01-01")
        except:
            time.sleep(5)
            return_str += daily_cal(code=code, begin_time="2023-01-01")
        return_str += '\n\n'

    return_str += '\n' + '震荡需要在压力位之中；趋势需要在走势通道中；周线如果是趋势状态，不要做逆趋势方向 \n'
    return_str += 'b. 缠轮判断方法，看图笔有没有改变方向，是否可能构成笔中枢 \n'

    return_str += '2. 趋势行情开仓策略（1买最好明确形成趋势即穿越ma60，2、3买、突破买）/ 震荡行情开仓策略（下档开仓，上档止盈）\n'
    return_str += '3. 设置好撤单前有效的止损，最重要的原则是不要亏损' + ' 4. 趋势平仓策略 / 震荡平仓策略'
    print(return_str)
    send_msg(return_str, type='text')
