#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 20:13
# @Author  : rockieluo
# @File    : online_trading_script.py

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict
from joblib import load

sys.path.append('/root/chan.py')

import numpy as np
import pytz
from tigeropen.common.util.contract_utils import future_contract
from tigeropen.common.util.order_utils import market_order, trail_order
from tigeropen.common.consts import SecurityType, Currency, Market

from DataAPI.TigerMockAPI import TigerMock
from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, TREND_TYPE
from ML.experiment_20240216_1.buy_data_generation import buy_stragety_feature as buy_stragety_feature
from Test.config import Config
from get_image_api import send_msg

fut_multiplier = {'MNQmain': 2, 'MRTYmain': 5, 'MYMmain': 0.5}
trailing_percent = 1
waiting_mins = 30
threshold = 0.3
retrace_rate = 0.003


def is_string_in_list(s, lst):
    return any(s in item for item in lst)


def predict_bsp(model, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = np.nan
    feature_arr = [missing] * len(meta)
    fea_list = []
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
            fea_list.append((feat_name, feat_value))
    feature_arr = [feature_arr]
    return model.predict_proba(feature_arr)[:, 1]


def exec_market_sell(sent_buy_order, open_orders, trade_client, client_config):
    contract = future_contract(symbol=sent_buy_order.contract.identifier, currency='USD')
    # 撤销原止损单
    trail_od_list = [od.id for od in open_orders if od.order_type == 'TRAIL']
    if len(trail_od_list) > 0:
        trail_oid = trail_od_list[0]
        trade_client.cancel_order(id=trail_oid)
    sell_order = market_order(account=client_config.account, contract=contract, action='SELL',
                              quantity=1)
    sell_oid = trade_client.place_order(sell_order)
    return sell_oid


def buy_model_predict(code, begin_time, end_time):
    send_msg(f'正常启动 {code} 程序', type='text')
    data_src = DATA_SRC.TigerMock
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
    )

    cur_path = os.path.dirname(os.path.realpath(__file__))
    model = load(f'{cur_path}/../experiment_20240216_1/buy_model_{code}_ma60_20240216_1.joblib')
    meta = json.load(open(f'{cur_path}/../experiment_20240216_1/buy_feature_{code}_ma60_20240216_1.meta', 'r'))

    for _ in chan.step_load():
        pass

    # 上面是初始化历史数据，发送消息初始化成功
    # 下面开始，每5s执行一次
    # 每天超过指定时间（统一使用美东时间，比较大小用timestamp），break掉，并发送消息
    # 目前持有非当期合约，平仓平
    # 目前无持仓且无未成交开仓订单：判断有没有一条全新的刚好走完的5min bar，有的话加入chan对象，判断是否开仓，开仓连带固定比例止损单也一起提交；不开仓则继续等；开仓则记录订单id
    # 目前无持仓但是有未成交订单：马上撤销所有
    # 目前有持仓且无市价卖出单：监控当前价格，如果价格濒临跌破成本，撤销原有止损单，提交一个市价卖出单；否则啥也不用做，等利润奔跑；
    # 目前有持仓且有市价卖出单：如果该市价卖出单超过5s则撤销掉，并发出告警，一般市价就是一定要成交的。
    # 目前有持仓却没有止损单
    # 目前有持仓，判断是否为空单，是的话赶紧平仓

    data_src_instance_5m = TigerMock(code, k_type=KL_TYPE.K_5M, begin_date=None, end_date=None)
    data_src_instance_1m = TigerMock(code, k_type=KL_TYPE.K_1M, begin_date=None, end_date=None)
    data_src_instance_5m.do_init()  # 初始化数据源类
    client_config = data_src_instance_5m.client_config
    trade_client = data_src_instance_5m.trade_client
    quote_client = data_src_instance_5m.quote_client
    today_contract = quote_client.get_current_future_contract(code.strip('main'))  # 获取当期交易的标的
    today_code = today_contract.loc[0, 'contract_code']

    # 初始化一些价格，应对跨天
    last_bi_begin_low = chan[0][-1][-1].close
    max_price = chan[0][-1][-1].high
    last_buy_ma60 = chan[0][-1][-1].trend[TREND_TYPE.MEAN][60]

    while True:
        time.sleep(6)
        # 获取当前UTC时间
        now_utc = datetime.now(pytz.utc)
        # 将当前UTC时间转换为美东时间
        eastern = pytz.timezone("US/Eastern")
        now_eastern = now_utc.astimezone(eastern)
        # 判断当前美东时间是否在17点到18点之间，如果是则停止运行
        if 17 <= now_eastern.hour < 18:
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序正常关闭", type='text')
            break

        # 查数据不断更新chan对象是一定要做的
        for klu_5m in data_src_instance_5m.get_kl_data(limit=2):
            if klu_5m.time <= chan[0][-1][-1].time:
                continue
            chan.trigger_load({KL_TYPE.K_5M: [klu_5m]})

        positions = trade_client.get_positions(sec_type=SecurityType.FUT, currency=Currency.ALL, market=Market.ALL)
        open_orders = trade_client.get_open_orders(sec_type=SecurityType.FUT, market=Market.ALL)
        open_orders = [od for od in open_orders if od.contract.identifier == today_code]
        filled_orders = trade_client.get_filled_orders(
            sec_type=SecurityType.FUT, market=Market.ALL,
            start_time=int(now_utc.timestamp() - 60 * 24 * 60 * 60) * 1000,
            end_time=int(now_utc.timestamp()) * 1000)
        filled_orders = [od for od in filled_orders if od.contract.identifier == today_code]
        sent_buy_order = max(filled_orders, key=lambda x: x.trade_time)
        last_bi = chan[0].bi_list[-1]

        if len(positions) > 0 and today_code in [pos.contract.identifier for pos in positions if pos.quantity > 0]:
            for last_klu in data_src_instance_1m.get_kl_data(limit=2):
                klu_1m = last_klu
            if klu_1m.high > max_price:
                max_price = klu_1m.high

        # 目前持有非当期合约，平仓
        if is_string_in_list(code[:3], [pos.contract.identifier for pos in positions]) and \
                today_code not in [pos.contract.identifier for pos in positions]:
            for pos in positions:
                if pos.contract.identifier != today_code and code[:3] in pos.contract.identifier:
                    contract = future_contract(symbol=pos.contract.identifier, currency='USD')
                    # 生成订单对象
                    wrong_ctrt_month_order = market_order(account=client_config.account, contract=contract,
                                                          action='SELL', quantity=pos.quantity)
                    wrong_ctrt_month_oid = trade_client.place_order(wrong_ctrt_month_order)
                    print(f'wrong month contract, sell. wrong_ctrt_month_oid = {wrong_ctrt_month_oid}')
                    send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序发现非当期合约，平仓",
                             type='text')

        # 目前有持仓，判断是否为空单，是的话赶紧平仓
        elif len(positions) > 0 and \
                today_code in [pos.contract.identifier for pos in positions if pos.quantity < 0]:
            for pos in positions:
                if today_code == pos.contract.identifier and pos.quantity < 0:
                    contract = future_contract(symbol=pos.contract.identifier, currency='USD')
                    # 生成订单对象，这里原有quantity是负的，要*-1，
                    close_order = market_order(account=client_config.account, contract=contract, action='BUY',
                                               quantity=-1 * pos.quantity)
                    close_oid = trade_client.place_order(close_order)
                    print(f'why you have short position? close id = {close_oid}')
                    send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序发现持有空单，平仓",
                             type='text')

        # 目前无持仓（没有任何标的或者该标的不在持仓标的中）但是有未成交的该标的订单：马上撤销所有
        elif (len(positions) == 0 or today_code not in [pos.contract.identifier for pos in positions]) \
                and len(open_orders) > 0:
            for od in open_orders:
                if today_code == od.contract.identifier:
                    trade_client.cancel_order(id=od.id)
                    print('cancel open order')
                    send_msg(
                        f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序撤销未成交订单，市价订单马上要成交",
                        type='text')

        # 目前无持仓且无未成交开仓订单：判断有没有一条全新的刚好走完的5min bar，有的话加入chan对象，判断是否开仓，
        # 开仓连带固定比例止损单也一起提交；不开仓则继续等
        elif (len(positions) == 0 or today_code not in [pos.contract.identifier for pos in positions]) \
                and len(open_orders) == 0:

            last_klu_5m = chan[0][-1][-1]
            last_last_klu_5m = [klu for ckl in chan[0][-2:] for klu in ckl][-2]
            bsp_list = [bsp for bsp in chan.get_bsp() if bsp.is_buy]
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]
            cur_lv_chan = chan[0]

            if last_last_klu_5m.trend[TREND_TYPE.MEAN][5] < last_last_klu_5m.trend[TREND_TYPE.MEAN][60] and \
                    last_klu_5m.trend[TREND_TYPE.MEAN][5] > last_klu_5m.trend[TREND_TYPE.MEAN][60]:
                last_bsp.features.add_feat(buy_stragety_feature(last_klu_5m, cur_lv_chan, bsp_list))  # 开仓K线特征
                pred_prob = predict_bsp(model, last_bsp, meta)[0]
            else:
                continue

            # print('pred prob is ', pred_prob)
            if pred_prob > threshold:
                contract = future_contract(symbol=today_code, currency='USD')
                # 生成订单对象
                buy_order = market_order(account=client_config.account, contract=contract, action='BUY', quantity=1)
                buy_oid = trade_client.place_order(buy_order)
                print(f'{cur_lv_chan[-1][-1].time}:buy id = {buy_oid}')

                last_buy_klc_idx = chan[0][-1].idx
                last_buy_price = last_klu_5m.close
                last_bi_begin_low = last_bi.get_begin_klu().low \
                    if last_buy_klc_idx in last_bi.klc_lst and last_bi.is_up() else min(
                    last_buy_price * 0.998, max(min([ckl.low for ckl in chan[0][-5:]]), last_buy_price - 50))
                last_buy_ma60 = last_klu_5m.trend[TREND_TYPE.MEAN][60]
                max_price = last_klu_5m.high

                # 移动止损订单
                trail_order_obj = trail_order(
                    account=client_config.account, contract=contract, action='SELL', quantity=1,
                    trailing_percent=trailing_percent)
                trail_oid = trade_client.place_order(trail_order_obj)
                print(f'{cur_lv_chan[-1][-1].time}:trail id = {trail_oid}')

                send_msg(
                    f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} ma60程序开仓，"
                    f"最新k线结束时间 {last_klu_5m.time.to_str()}",
                    type='text')

        # 目前有持仓且无市价卖出单：监控当前价格，如果价格跌破重要止损位，撤销原有止损单，提交一个市价卖出单；否则啥也不用做，等利润奔跑；
        elif len(positions) > 0 and \
                today_code in [pos.contract.identifier for pos in positions if pos.quantity > 0] and \
                'MKT' not in [od.order_type for od in open_orders] and klu_1m.close < last_bi_begin_low:
            sell_oid = exec_market_sell(sent_buy_order, open_orders, trade_client, client_config)
            print(f'跌破重要止损位，止损卖出, sell id {sell_oid}')
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} ma60程序跌破重要止损位",
                     type='text')

        # 回撤止盈卖出
        elif len(positions) > 0 and \
                today_code in [pos.contract.identifier for pos in positions if pos.quantity > 0] and \
                'MKT' not in [od.order_type for od in open_orders] and klu_1m.low > last_buy_ma60 and \
                (max_price - klu_1m.low) / max_price > retrace_rate and klu_1m.close < klu_1m.open:
            sell_oid = exec_market_sell(sent_buy_order, open_orders, trade_client, client_config)
            print(f'回撤止盈卖出, sell id {sell_oid}')
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} ma60程序回撤止盈卖出",
                     type='text')

        # ma60死叉止盈卖出
        elif len(positions) > 0 and \
                today_code in [pos.contract.identifier for pos in positions if pos.quantity > 0] and \
                'MKT' not in [od.order_type for od in open_orders] and \
                sent_buy_order.trade_time / 1000 + waiting_mins * 60 < int(datetime.now(pytz.utc).timestamp()) and \
                chan[0][-1][-1].trend[TREND_TYPE.MEAN][5] < chan[0][-1][-1].trend[TREND_TYPE.MEAN][60] and \
                klu_1m.close > sent_buy_order.avg_fill_price:
            sell_oid = exec_market_sell(sent_buy_order, open_orders, trade_client, client_config)
            print(f'ma60死叉止盈卖出, sell id {sell_oid}')
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} ma60程序 ma60死叉止盈卖出",
                     type='text')

        # 目前有该标的持仓且有市价卖出单：如果该市价卖出单超过5s则撤销掉，并发出告警，一般市价就是一定要成交的。
        elif len(positions) > 0 and today_code in [pos.contract.identifier for pos in positions] and \
                'MKT' in [od.order_type for od in open_orders]:
            for od in open_orders:
                if today_code == od.contract.identifier and od.order_type == 'MKT':
                    trade_client.cancel_order(id=od.id)
                    print('has position and mkt order not filled, cancel open order')
                    send_msg(
                        f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序发现市价止损卖出单没有正常成交，撤销它",
                        type='text')

        # 目前有持仓却没有止损单
        elif len(positions) > 0 and \
                today_code in [pos.contract.identifier for pos in positions if pos.quantity > 0] and \
                'TRAIL' not in [od.order_type for od in open_orders]:
            contract = future_contract(symbol=today_code, currency='USD')
            # 移动止损订单
            new_trail_order_obj = trail_order(
                account=client_config.account, contract=contract, action='SELL', quantity=1,
                trailing_percent=trailing_percent)
            new_trail_oid = trade_client.place_order(new_trail_order_obj)
            print(f'have position but no trail order, place order trail id = {new_trail_oid}')
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序发现有持仓却没有止损单",
                     type='text')

        else:
            print(f'why you can enter here?')
            send_msg(f"美东时间 {now_eastern.strftime('%Y-%m-%d %H:%M:%S')} {code} 程序发现进入其他未知情况分支",
                     type='text')


if __name__ == '__main__':
    # 更改导入的特征计算模块、模型文件、特征文件
    # 实盘设置开盘前5分钟启动，这里end_time设置为None，会自动拉最新的1000条数据来初始化
    fut_code = 'MNQmain'
    # '2024-01-19 16:50:00'
    try:
        buy_model_predict(code=fut_code, begin_time=None, end_time=None)
    except Exception as e:
        send_msg(f"{str(e)} 出错了，请检查！", type='text')
