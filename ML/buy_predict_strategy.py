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
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from ML.buy_data_generation import buy_stragety_feature
from Plot.PlotDriver import CPlotDriver
from Test.config import Config
from get_image_api import send_msg, get_token, upload_image



def kelly_cangwei(p):
    """
    f = p — q 就是最优的下注比例，它就是凯利公式。假设每局的赔率等于 1，赢了翻倍，输了亏光

    :return:
    """
    return p - (1 - p)


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


def buy_model_predict(code, begin_time, only_bsp, is_send):
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    end_time = None
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
    model.load_model(f"buy_model_{code}.json")
    meta = json.load(open(f"buy_feature_{code}.meta", "r"))

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
        if only_bsp:
            if (cur_lv_chan[-3].idx != last_bsp.klu.klc.idx or last_bsp.klu.idx in treated_bsp_idx or not last_bsp.is_buy) and is_hold is False:
                # and cur_lv_chan[-2].idx != last_bsp.klu.klc.idx)
                    # 已经判断过了，分型还没形成，不是买点
                    continue
        # if not last_bsp.is_buy:
        #     continue

        last_bsp.features.add_feat(buy_stragety_feature(last_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
        # 买卖点打分，应该和demo5最后的predict结果完全一致才对
        # print(last_bsp.klu.time, last_klu.time.to_str())

        pred_prob = predict_bsp(model, last_bsp, meta)[0]

        prob_dict['bsp_time'].append(last_bsp.klu.time)
        prob_dict['last_klu_time'].append(last_klu.time)
        prob_dict['prob'].append(pred_prob)
        prob_dict['bsp_type'].append(last_bsp.is_buy)
        treated_bsp_idx.add(last_bsp.klu.idx)
        if pred_prob > 0.5 and is_hold is False:
            plot_marker[last_klu.time.to_str()] = (
                pred_prob, "down" if last_bsp.is_buy else "up")
            last_buy_price = cur_lv_chan[-1][-1].close
            print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            max_price = cur_lv_chan[-1][-1].close
            last_buy_time = cur_lv_chan[-1][-1].time
            is_hold = True
            hold_indx = 0
            trade_info['buy_time'].append(cur_lv_chan[-1][-1].time.to_str())
            trade_info['buy_price'].append(last_buy_price)

        stop_loss_retrace_rate = 0.2
        stop_loss_diff = 0.5
        if is_hold and cur_lv_chan[-1][-1].time > last_buy_time:
            hold_indx += 1
            if cur_lv_chan[-1][-1].high > max_price:
                max_price = cur_lv_chan[-1][-1].high
            if ((max_price - cur_lv_chan[-1][-1].low)*100/(max_price) > stop_loss_retrace_rate and pred_prob < 0.5 and \
                    cur_lv_chan[-1][-1].close < cur_lv_chan[-1][-1].open) or (hold_indx > 2 and cur_lv_chan[-1][-1].close < last_buy_price + stop_loss_diff):  # 阳线的价差也比较大，但是大阳线不应该卖
                if (hold_indx > 2 and cur_lv_chan[-1][-1].close < last_buy_price + stop_loss_diff) and not ((max_price - cur_lv_chan[-1][-1].low)*100/(max_price) > stop_loss_retrace_rate and pred_prob < 0.5 and \
                    cur_lv_chan[-1][-1].close < cur_lv_chan[-1][-1].open):
                    sell_price = cur_lv_chan[-1][-1].close
                    sell_reason = '15 min quit'
                else:
                    sell_price = np.round(max_price * (1-stop_loss_retrace_rate/100), 2)
                    sell_reason = 'retrace from max'
                print(
                    f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, '
                    f'profit rate = {(sell_price - last_buy_price) / last_buy_price * 100:.2f}%')
                trade_info['sell_time'].append(cur_lv_chan[-1][-1].time.to_str())
                trade_info['sell_price'].append(sell_price)
                trade_info['max_price'].append(max_price)
                trade_info['sell_reason'].append(sell_reason)
                trade_info['profit'].append((sell_price - last_buy_price) / last_buy_price * 100)
                trade_info['real_profit'].append((sell_price - last_buy_price) * 5 - 2.8)
                is_hold = False
            # 这里的时间是该bar的开始时间，futu显示的结束时间，所以futu慢5min

    if len(trade_info['buy_time']) > len(trade_info['sell_time']):
        trade_info['buy_time'] = trade_info['buy_time'][:-1]
        trade_info['buy_price'] = trade_info['buy_price'][:-1]
    trade_df = pd.DataFrame(trade_info)
    df_sorted = trade_df.sort_values('profit')
    # 去掉 'A' 列最低和最高的两行
    df_rm_highest_lowest = df_sorted.iloc[1:-1]
    mean_profit = np.mean(df_rm_highest_lowest.real_profit)
    print(f"去掉最高最低的交易，平均每笔交易盈利{mean_profit: .2f}刀")

    # 计算交易胜率
    winning_trades = trade_df[trade_df['profit'] > 0]
    win_rate = len(winning_trades) / len(trade_df)
    print(f"交易胜率: {win_rate * 100:.2f}%")

    # 计算夏普比率
    sharpe_ratio = np.mean(trade_df['profit']) / np.std(trade_df['profit'])
    print(f"夏普比率: {sharpe_ratio:.2f}")

    # 计算平均每天交易次数
    trading_days = (pd.to_datetime(trade_df['sell_time'], format='%Y/%m/%d %H:%M').max() -
                    pd.to_datetime(trade_df['buy_time'], format='%Y/%m/%d %H:%M').min()).days + 1
    average_daily_trades = len(trade_df) / trading_days
    print(f"平均每天交易次数: {average_daily_trades:.2f}")

    df = pd.DataFrame(prob_dict)
    df = df.sort_values('last_klu_time')
    df['prob'] = df['prob'].apply(lambda x: f"{x:.3f}")
    if not only_bsp:
        for index, row in df.tail(1).iterrows():
            row_str = ', '.join([f"{col_name}: {col_value}" for col_name, col_value in row.items()])
        print(f"日常预测上涨概率: ", row_str)
        cangwei = kelly_cangwei(float(df['prob'].iloc[-1]))
        if cangwei < 0:
            cangwei = '注意风险'
        all_msg_to_send = f"{code} 日常预测上涨概率: " + row_str + f", 做多仓位{cangwei}"
    else:
        print("近期买点成立概率: ")
        all_msg_to_send = f"{code} 近期买点成立概率: "
        for index, row in df.tail(2).iterrows():
            row_str = ', '.join([f"{col_name}: {col_value}" for col_name, col_value in row.items()])
            print(f"{row_str}")
            all_msg_to_send += f"\n {row_str}"
        plot(chan, plot_marker)

    if is_send:
        send_msg(all_msg_to_send, type='text')
        if only_bsp:
            access_token = get_token()
            res = upload_image('buy_label.png', access_token)
            res = json.loads(res)
            if res['code'] == 0:
                send_msg(res['data']['image_key'], type='image')

    feature_importance = model.get_score(importance_type='weight')
    print(feature_importance)


if __name__ == '__main__':
    for code in ['MRTYmain']:
        # try:
        #     buy_model_predict(code=code, begin_time="2022-01-01", only_bsp=False, is_send=True)
        # except:
        #     time.sleep(5)
        #     buy_model_predict(code=code, begin_time="2022-01-01", only_bsp=False, is_send=True)
        # try:
        buy_model_predict(code=code, begin_time="2023-12-01 00:00:00", only_bsp=True, is_send=False)
        # except:
        #     time.sleep(5)
        #     buy_model_predict(code=code, begin_time="2022-01-01", only_bsp=True, is_send=True)
