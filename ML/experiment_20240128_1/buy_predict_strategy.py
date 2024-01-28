#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 17:34
# @Author  : rockieluo
# @File    : buy_predict_strategy.py


import json
import sys
import time
from typing import Dict, TypedDict

import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.append('/root/chan.py')

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, FX_TYPE
from Common.CTime import CTime
from ML.experiment_20240128_1.buy_data_generation import buy_stragety_feature
from Plot.PlotDriver import CPlotDriver
from Test.config import Config


fut_multiplier = {'MNQmain': 2, 'MRTYmain': 5, 'MYMmain': 0.5}


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
    # print(fea_list)
    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)
    return model.predict(dtest)


def buy_model_predict(code, begin_time, retrace_rate, wait_time_indx, end_time=None):
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    # end_time = None
    data_src = DATA_SRC.CSV
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

    model = xgb.Booster()
    model.load_model(f"buy_model_{code}_20240128_1.json")
    meta = json.load(open(f"buy_feature_{code}_20240128_1.meta", "r"))

    treated_bsp_idx = set()
    prob_dict = {'bsp_time': [], 'last_klu_time': [], 'prob': [], 'bsp_type': []}
    plot_marker = {}
    is_hold = False
    trade_info = {'sell_reason': [], 'buy_time': [], 'buy_price': [],  'sell_time':[], 'sell_price': [], 'max_price': [], 'profit':[], 'real_profit': []}
    begin_not_trade_indx = 0
    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        if begin_not_trade_indx < 200:
            begin_not_trade_indx += 1
            continue

        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]

        # 没有持有，要进入判断是否买入，不能continue
        # 有持有，要进入判断是否卖出，不能continue

        # if not last_bsp.is_buy:
        #     continue

        if is_hold is False and cur_lv_chan[-3].idx == last_bsp.klu.klc.idx:
            if last_bsp.klu.idx in treated_bsp_idx or not last_bsp.is_buy:
                continue

            last_bsp.features.add_feat(buy_stragety_feature(last_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
            # 买卖点打分，应该和demo5最后的predict结果完全一致才对
            # print(last_bsp.klu.time, last_klu.time.to_str())

            pred_prob = predict_bsp(model, last_bsp, meta)[0]
            treated_bsp_idx.add(last_bsp.klu.idx)
            if pred_prob < 0.6:
                continue

            plot_marker[last_klu.time.to_str()] = (
                pred_prob, "down" if last_bsp.is_buy else "up")
            last_buy_price = cur_lv_chan[-1][-1].close
            # print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            max_price = cur_lv_chan[-1][-1].close
            max_price_ckl = cur_lv_chan[-1]
            last_buy_time = cur_lv_chan[-1][-1].time
            is_hold = True
            hold_indx = 0
            trade_info['buy_time'].append(cur_lv_chan[-1][-1].time.to_str())
            trade_info['buy_price'].append(last_buy_price)

        stop_loss_diff = 2.7 / fut_multiplier[code]
        # hold_indx > 2，说明持仓超过10min，达15min
        # hold_indx > 3，说明持仓超过15min，达20min
        if is_hold and cur_lv_chan[-1][-1].time > last_buy_time:
            hold_indx += 1
            if cur_lv_chan[-1][-1].high > max_price:
                max_price = cur_lv_chan[-1][-1].high
                max_price_ckl = cur_lv_chan[-1]

            # 阳线的价差也比较大，但是大阳线不应该卖
            if ((max_price - cur_lv_chan[-1][-1].low) * 100 / max_price > retrace_rate and
                cur_lv_chan[-1][-1].close < cur_lv_chan[-1][-1].open) \
                    or \
                    (hold_indx > wait_time_indx and cur_lv_chan[-1][-1].low < last_buy_price + stop_loss_diff):
                if (hold_indx > wait_time_indx and cur_lv_chan[-1][-1].low < last_buy_price + stop_loss_diff) and \
                        not ((max_price - cur_lv_chan[-1][-1].low) * 100 / max_price > retrace_rate and
                             cur_lv_chan[-1][-1].close < cur_lv_chan[-1][-1].open):
                    sell_price = cur_lv_chan[-1][-1].close
                    sell_reason = 'wait time quit'
                else:
                    sell_price = np.round(max_price * (1-retrace_rate/100), 2)
                    sell_reason = 'retrace from max'
                # print(
                #     f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, '
                #     f'profit rate = {(sell_price - last_buy_price) / last_buy_price * 100:.2f}%')
                trade_info['sell_time'].append(cur_lv_chan[-1][-1].time.to_str())
                trade_info['sell_price'].append(sell_price)
                trade_info['max_price'].append(max_price)
                # trade_info['fx_time'] = max_price_ckl[-1].time.to_str()
                trade_info['sell_reason'].append(sell_reason)
                trade_info['profit'].append((sell_price - last_buy_price) / last_buy_price * 100)
                trade_info['real_profit'].append((sell_price - last_buy_price) * fut_multiplier[code] - 2.8)
                is_hold = False
            # 这里的时间是该bar的开始时间，futu显示的结束时间，所以futu慢5min


    if len(trade_info['buy_time']) > len(trade_info['sell_time']):
        trade_info['buy_time'] = trade_info['buy_time'][:-1]
        trade_info['buy_price'] = trade_info['buy_price'][:-1]
    trade_df = pd.DataFrame(trade_info)
    df_sorted = trade_df.sort_values('real_profit')
    # 去掉 'A' 列最低和最高的两行
    df_rm_highest_lowest = df_sorted.iloc[1:-1]
    mean_profit = np.mean(df_rm_highest_lowest.real_profit)
    print(f"去掉最高最低的交易，平均每笔交易盈利{mean_profit: .2f}刀")

    total_profit = np.sum(df_rm_highest_lowest.real_profit)
    print(f"去掉最高最低的交易，总交易盈利{total_profit: .2f}刀")

    # 计算交易胜率
    winning_trades = trade_df[trade_df['real_profit'] > 0]
    win_rate = len(winning_trades) / len(trade_df)
    print(f"交易胜率: {win_rate * 100:.2f}%")

    # 计算夏普比率
    sharpe_ratio = np.mean(trade_df['real_profit']) / np.std(trade_df['real_profit'])
    print(f"夏普比率: {sharpe_ratio:.2f}")

    # 计算平均每天交易次数
    trading_days = (pd.to_datetime(trade_df['sell_time'], format='%Y/%m/%d %H:%M', errors='coerce').max() -
                    pd.to_datetime(trade_df['buy_time'], format='%Y/%m/%d %H:%M', errors='coerce').min()).days + 1
    average_daily_trades = len(trade_df) / trading_days
    print(f"平均每天交易次数: {average_daily_trades:.2f}")

    # 计算盈利交易的平均盈利
    average_profit = df_rm_highest_lowest[df_rm_highest_lowest['real_profit'] > 0]['real_profit'].mean()
    # 计算亏损交易的平均亏损（取绝对值）
    average_loss = abs(df_rm_highest_lowest[df_rm_highest_lowest['real_profit'] < 0]['real_profit'].mean())
    # 计算赔率
    odds = average_profit / average_loss
    print("赔率：", odds)

    # 预期收益率
    exp_return = win_rate / (1 - win_rate) * odds
    print("预期收益率: ", exp_return)

    # feature_importance = model.get_score(importance_type='weight')
    # print(feature_importance)

    return exp_return, odds, win_rate, average_daily_trades, mean_profit, total_profit


if __name__ == '__main__':
    # 记得改import中特征计算模块的导入
    res_dict = {'retrace_rate': [], 'wait_time_indx': [], 'exp_return': [], 'odds':[], 'win_rate':[], 'average_daily_trades': [], 'mean_profit': [], 'total_profit': []}
    for code in ['MNQmain']:
        # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
        for retrace_rate in [0.1]:
            # 0, 1, 2, 3, 4, 5, 6
            for wait_time_indx in [6]:
                exp_return, odds, win_rate, average_daily_trades, mean_profit, total_profit = buy_model_predict(code=code, begin_time="2023-11-01 00:00:00", end_time="2024-01-18 00:00:00", retrace_rate=retrace_rate, wait_time_indx=wait_time_indx)
                print('retrace_rate is ', retrace_rate, ', wait_time_indx is ', wait_time_indx, ', exp_return is ', exp_return)
                res_dict['retrace_rate'].append(retrace_rate)
                res_dict['wait_time_indx'].append(wait_time_indx)
                res_dict['exp_return'].append(exp_return)
                res_dict['odds'].append(odds)
                res_dict['win_rate'].append(win_rate)
                res_dict['average_daily_trades'].append(average_daily_trades)
                res_dict['mean_profit'].append(mean_profit)
                res_dict['total_profit'].append(total_profit)
    res_df = pd.DataFrame(res_dict)
    res_df = res_df.sort_values('exp_return')
    print(res_df)
