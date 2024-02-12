import json
import os
from datetime import datetime
from typing import Dict, TypedDict, List
from joblib import load

import pandas as pd
import numpy as np
import xgboost as xgb

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, KL_TYPE
from DataAPI.BaoStockAPI import CBaoStock
from DataAPI.csvAPI import CSV_API
from KLine.KLine_Unit import CKLine_Unit
from ML.experiment_20240209_5.buy_data_generation import buy_stragety_feature as buy_stragety_feature3
from ML.experiment_20240209_4.buy_data_generation import buy_stragety_feature as buy_stragety_feature4
from ML.experiment_20240209_6.buy_data_generation import buy_stragety_feature as buy_stragety_feature5
from Test.config import Config

fut_multiplier = {'MNQmain': 2, 'MRTYmain': 5, 'MYMmain': 0.5}


def predict_bsp(model, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = np.nan
    # missing = 0
    feature_arr = [missing] * len(meta)
    fea_list = []
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
            fea_list.append((feat_name, feat_value))
    feature_arr = [feature_arr]
    # print(feature_arr)
    return model.predict_proba(feature_arr)[:, 1]


def combine_5m_klu_form_1m(klu_1m_lst: List[CKLine_Unit]) -> CKLine_Unit:
    return CKLine_Unit(
        {
            DATA_FIELD.FIELD_TIME: klu_1m_lst[-1].time,
            DATA_FIELD.FIELD_OPEN: klu_1m_lst[0].open,
            DATA_FIELD.FIELD_CLOSE: klu_1m_lst[-1].close,
            DATA_FIELD.FIELD_HIGH: max(klu.high for klu in klu_1m_lst),
            DATA_FIELD.FIELD_LOW: min(klu.low for klu in klu_1m_lst),
            DATA_FIELD.FIELD_VOLUME: sum([klu.trade_info.metric['volume'] for klu in klu_1m_lst]),
        }
    )


def buy_model_predict(code, begin_time, dependent_time, threshold, retrace_rate, wait_time_seconds, end_time=None):
    data_src_type = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_5M]  #, KL_TYPE.K_1M

    config_object = Config()
    chan_config = config_object.read_chan_config_trigger_step
    config = CChanConfig(chan_config)

    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=lv_list,
        config=config,
        is_combine=True
    )

    cur_path = os.path.dirname(os.path.realpath(__file__))
    model3 = load(f'{cur_path}/../experiment_20240209_5/buy_model_{code}_bsp3_20240209_5.joblib')
    meta3 = json.load(open(f'{cur_path}/../experiment_20240209_5/buy_feature_{code}_bsp3_20240209_5.meta', 'r'))

    model4 = load(f'{cur_path}/../experiment_20240209_4/buy_model_{code}_bsp4_20240209_4.joblib')
    meta4 = json.load(open(f'{cur_path}/../experiment_20240209_4/buy_feature_{code}_bsp4_20240209_4.meta', 'r'))

    model5 = load(f'{cur_path}/../experiment_20240209_6/buy_model_{code}_bsp5_20240209_6.joblib')
    meta5 = json.load(open(f'{cur_path}/../experiment_20240209_6/buy_feature_{code}_bsp5_20240209_6.meta', 'r'))

    data_src_1m = CSV_API(code, k_type=KL_TYPE.K_1M, begin_date=dependent_time, end_date=end_time, autype=AUTYPE.QFQ)

    is_hold = False
    trade_info = {'sell_reason': [], 'which_klu_is_bsp': [], 'buy_time': [], 'buy_price': [], 'sell_time': [], 'sell_price': [],
                  'max_price': [], 'profit': [], 'real_profit': []}
    treated_bsp_idx = set()
    treated_1m_time = set()
    klu_1m_lst_tmp: List[CKLine_Unit] = []
    sell_price = 0
    sell_reason = ''
    for last_klu_1m in data_src_1m.get_kl_data():
        klu_1m_lst_tmp.append(last_klu_1m)
        if len(klu_1m_lst_tmp) == 5:  # 已经完成5根1分钟K线了，说明这个最新的5分钟K线和里面的5根1分钟K线在将来不会再变化
            klu_5m = combine_5m_klu_form_1m(klu_1m_lst_tmp)  # 合成60分钟K线
            chan.trigger_load({KL_TYPE.K_5M: [klu_5m]})  #, KL_TYPE.K_1M: klu_1m_lst_tmp
            klu_1m_lst_tmp = []

            # if len([klu.time.to_str() for klu in chan[0].klu_iter()]) < 1300:
            #     continue
            dt = datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S")
            timestamp = int(dt.timestamp())
            if [klu.time for klu in chan[0].klu_iter()][-1].ts < timestamp:
                continue

            last_5m_klu = chan[0][-1][-1]
            bsp_list = chan.get_bsp(idx=0)
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]
            cur_5m_chan = chan[0]
            # print([klu.time.to_str() for klu in cur_5m_chan[-1]])

            if is_hold is False and (cur_5m_chan[-3].idx == last_bsp.klu.klc.idx or cur_5m_chan[-4].idx == last_bsp.klu.klc.idx):
                if not last_bsp.is_buy:
                    # last_bsp.klu.idx in treated_bsp_idx or
                    continue

                if cur_5m_chan[-3].idx == last_bsp.klu.klc.idx:
                    last_bsp.features.add_feat(buy_stragety_feature3(last_5m_klu, cur_5m_chan, bsp_list))  # 开仓K线特征
                    pred_prob = predict_bsp(model3, last_bsp, meta3)[0]
                    treated_bsp_idx.add(last_bsp.klu.idx)
                    print(last_bsp.klu.time.to_str(), np.round(pred_prob, 3))
                    if pred_prob < threshold:
                        continue
                    which_klu_is_bsp = 3
                elif cur_5m_chan[-4].idx == last_bsp.klu.klc.idx:
                    last_bsp.features.add_feat(buy_stragety_feature4(last_5m_klu, cur_5m_chan, bsp_list))  # 开仓K线特征
                    pred_prob = predict_bsp(model4, last_bsp, meta4)[0]
                    treated_bsp_idx.add(last_bsp.klu.idx)
                    print(last_bsp.klu.time.to_str(), np.round(pred_prob, 3))
                    if pred_prob < threshold:
                        continue
                    which_klu_is_bsp = 4
                elif cur_5m_chan[-5].idx == last_bsp.klu.klc.idx:
                    last_bsp.features.add_feat(buy_stragety_feature5(last_5m_klu, cur_5m_chan, bsp_list))  # 开仓K线特征
                    pred_prob = predict_bsp(model5, last_bsp, meta5)[0]
                    treated_bsp_idx.add(last_bsp.klu.idx)
                    print(last_bsp.klu.time.to_str(), np.round(pred_prob, 3))
                    if pred_prob < threshold:
                        continue
                    which_klu_is_bsp = 5

                last_buy_price = cur_5m_chan[-1][-1].close
                print(f'{cur_5m_chan[-1][-1].time}:buy price = {last_buy_price}, 最后一个klc的时间：{[klu.time.to_str() for klu in cur_5m_chan[-1]]}, 前几个klc的时间: {[klu.time.to_str() for klu in cur_5m_chan[-2]]}，{[klu.time.to_str() for klu in cur_5m_chan[-3]]}， {[klu.time.to_str() for klu in cur_5m_chan[-4]]} ')
                max_price = cur_5m_chan[-1][-1].close
                last_buy_time = cur_5m_chan[-1][-1].time
                last_buy_bi_idx = cur_5m_chan.bi_list[-1].idx
                last_buy_bi_begin_time = cur_5m_chan.bi_list[-1].get_begin_klu().time
                last_buy_klc_idx = cur_5m_chan[-1].idx
                is_hold = True
                trade_info['buy_time'].append(cur_5m_chan[-1][-1].time.to_str())
                trade_info['buy_price'].append(last_buy_price)
                trade_info['which_klu_is_bsp'].append(which_klu_is_bsp)

        # 检查时间对齐
        # print("当前所有5m K线:", [klu.time.to_str() for klu in chan[0].klu_iter()])
        # print("当前所有1M K线:", [klu.time.to_str() for klu in chan[1].klu_iter()], "\n")

        stop_loss_diff = 2.7 / fut_multiplier[code]
        if is_hold and last_klu_1m.time > last_buy_time:
            if last_klu_1m.time < last_buy_time or last_klu_1m.time.to_str() in treated_1m_time:
                continue
            treated_1m_time.add(last_klu_1m.time)
            if last_klu_1m.high > max_price:
                max_price = last_klu_1m.high

            last_bi = chan[0].bi_list[-1]

            # 无法覆盖一种情况是，该bar最终收涨，但实际上先到最高价再到最低价，会被甩出去
            if (max_price - last_klu_1m.low) / max_price > retrace_rate and \
                    last_klu_1m.close < last_klu_1m.open:
                sell_price = np.round(max_price * (1 - retrace_rate), 2)
                sell_reason = 'retrace from max'
                is_hold = False

            elif last_klu_1m.time.ts - last_buy_time.ts > wait_time_seconds and \
                    last_klu_1m.low < last_buy_price + stop_loss_diff:
                sell_price = last_klu_1m.close
                sell_reason = 'wait time quit'
                is_hold = False

            elif not last_bi.is_up() and last_bi.idx > last_buy_bi_idx:
                sell_price = last_klu_1m.close
                sell_reason = 'bi change'
                is_hold = False

            elif not last_bi.is_up() and last_buy_klc_idx in [klc.idx for klc in last_bi.klc_lst]:
                sell_price = last_klu_1m.close
                sell_reason = 'current bi change'
                is_hold = False

            if not is_hold:
                print('sell at ', last_klu_1m.time.to_str())
                trade_info['sell_time'].append(last_klu_1m.time.to_str())
                trade_info['sell_price'].append(sell_price)
                trade_info['max_price'].append(max_price)
                trade_info['sell_reason'].append(sell_reason)
                trade_info['profit'].append((sell_price - last_buy_price) / last_buy_price * 100)
                trade_info['real_profit'].append((sell_price - last_buy_price) * fut_multiplier[code] - 2.8)

    if len(trade_info['buy_time']) > len(trade_info['sell_time']):
        trade_info['buy_time'] = trade_info['buy_time'][:-1]
        trade_info['buy_price'] = trade_info['buy_price'][:-1]
    trade_df = pd.DataFrame(trade_info)
    df_sorted = trade_df.sort_values('real_profit')
    # 去掉 'A' 列最低和最高的两行
    df_rm_highest_lowest = df_sorted.iloc[1:-1]
    mean_profit = np.mean(trade_df.real_profit)
    print(f"平均每笔交易盈利{mean_profit: .2f}刀")

    total_profit = np.sum(trade_df.real_profit)
    print(f"总交易盈利{total_profit: .2f}刀")

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
    trade_times = len(trade_df)
    # print(f"平均每天交易次数: {average_daily_trades:.2f}")
    print(f"总交易次数: {len(trade_df):.2f}")

    # 计算盈利交易的平均盈利
    average_profit = trade_df[trade_df['real_profit'] > 0]['real_profit'].mean()
    # 计算亏损交易的平均亏损（取绝对值）
    average_loss = abs(trade_df[trade_df['real_profit'] < 0]['real_profit'].mean())
    # 计算赔率
    odds = average_profit / average_loss
    print("赔率：", odds)

    # 预期收益率
    exp_return = win_rate / (1 - win_rate) * odds
    print("预期收益率: ", exp_return)

    # feature_importance = model.get_score(importance_type='weight')
    # print(feature_importance)

    return exp_return, odds, win_rate, average_daily_trades, mean_profit, total_profit, trade_times


if __name__ == '__main__':
    # 记得改import中特征计算模块的导入
    res_dict = {'threshold': [], 'retrace_rate': [], 'wait_time_seconds': [], 'exp_return': [], 'odds': [], 'win_rate': [],
                'trade_times': [], 'mean_profit': [], 'total_profit': []}
    for code in ['MNQmain']:
        # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
        # 0.5, 0.6, 0.7
        for threshold in [0.5, 0.6, 0.7]:
            for retrace_rate in [0.0015]:
                # [15 * 60]
                for wait_time_min in [15 * 60]:
                    wait_time_seconds = wait_time_min * 60
                    # begin_time = "2023-11-01 00:01:00", end_time = "2023-11-31 00:00:00", dependent_time = "2023-10-01 00:01:00",
                    # begin_time = "2023-12-01 00:01:00", end_time = "2024-01-01 00:00:00", dependent_time = "2023-11-01 00:01:00",
                    # begin_time = "2024-01-01 00:01:00", end_time = "2024-01-31 00:00:00", dependent_time = "2023-12-01 00:01:00",
                    exp_return, odds, win_rate, average_daily_trades, mean_profit, total_profit, trade_times = buy_model_predict(
                        code=code, begin_time = "2023-12-01 00:01:00", end_time = "2024-01-31 00:00:00", dependent_time = "2023-11-28 00:01:00",
                        threshold=threshold, retrace_rate=retrace_rate, wait_time_seconds=wait_time_seconds)
                    print('threshold is ', threshold, 'retrace_rate is ', retrace_rate, ', wait_time_seconds is ',
                          wait_time_seconds, ', exp_return is ', exp_return)
                    res_dict['threshold'].append(threshold)
                    res_dict['retrace_rate'].append(retrace_rate)
                    res_dict['wait_time_seconds'].append(wait_time_seconds)
                    res_dict['exp_return'].append(exp_return)
                    res_dict['odds'].append(odds)
                    res_dict['win_rate'].append(win_rate)
                    res_dict['trade_times'].append(trade_times)
                    res_dict['mean_profit'].append(mean_profit)
                    res_dict['total_profit'].append(total_profit)
    res_df = pd.DataFrame(res_dict)
    res_df = res_df.sort_values('exp_return')
    print(res_df)
