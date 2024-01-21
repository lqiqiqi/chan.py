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
from typing import Dict, TypedDict

import pandas as pd
import numpy as np
import pytz
import xgboost as xgb
from tigeropen.common.util.contract_utils import stock_contract, future_contract
from tigeropen.common.util.order_utils import market_order, trail_order
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.consts import SecurityType, Currency, Market
from tigeropen.trade.trade_client import TradeClient

from DataAPI.TigerAPI import Tiger

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


def is_string_in_list(s, lst):
    return any(s in item for item in lst)


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


def buy_model_predict(code, begin_time, end_time, only_bsp, is_send):
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    data_src = DATA_SRC.Tiger
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

    model = xgb.Booster()
    model.load_model(f"buy_model_{code}.json")
    meta = json.load(open(f"buy_feature_{code}.meta", "r"))

    treated_bsp_idx = set()

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

    data_src_instance = Tiger(code, k_type=KL_TYPE.K_5M, begin_date=None, end_date=None)
    data_src_instance.do_init()  # 初始化数据源类
    client_config = data_src_instance.client_config
    trade_client = data_src_instance.trade_client
    quote_client = data_src_instance.quote_client
    today_contract = quote_client.get_current_future_contract(code.strip('main'))  # 获取当期交易的标的
    today_code = today_contract.loc[0, 'contract_code']

    while True:
        time.sleep(5)
        # 获取当前UTC时间
        now_utc = datetime.now(pytz.utc)
        # 将当前UTC时间转换为美东时间
        eastern = pytz.timezone("US/Eastern")
        now_eastern = now_utc.astimezone(eastern)
        # 判断当前美东时间是否在17点到18点之间，如果是则停止运行
        if 17 <= now_eastern.hour < 18:
            break

        # 查数据不断更新chan对象是一定要做的
        for klu in data_src_instance.get_kl_data(limit=2):
            if klu.time <= chan[0][-1][-1].time:
                continue
            chan.trigger_load({KL_TYPE.K_5M: [klu]})

        positions = trade_client.get_positions(sec_type=SecurityType.FUT, currency=Currency.ALL, market=Market.ALL)
        open_orders = trade_client.get_open_orders(sec_type=SecurityType.FUT, market=Market.ALL)
        open_orders = [od for od in open_orders if od.contract.symbol == today_code]
        filled_orders = trade_client.get_filled_orders(
            sec_type=SecurityType.FUT, market=Market.ALL, start_time=int(now_utc.timestamp() - 7 * 24 * 60 * 60)*1000,
            end_time=int(now_utc.timestamp())*1000)
        filled_orders = [od for od in filled_orders if od.contract.symbol == today_code]

        # 目前持有非当期合约，平仓
        if is_string_in_list(code[:4], [pos.contract.symbol for pos in positions]) and \
            today_code not in [pos.contract.symbol for pos in positions]:
            for pos in positions:
                if pos.contract.symbol != today_code:
                    contract = future_contract(symbol=pos.contract.symbol, currency='USD')
                    # 生成订单对象
                    wrong_ctrt_month_order = market_order(account=client_config.account, contract=contract,
                                                          action='SELL', quantity=pos.quantity)
                    wrong_ctrt_month_oid = trade_client.place_order(wrong_ctrt_month_order)
                    print(f'wrong month contract, sell. wrong_ctrt_month_oid = {wrong_ctrt_month_oid}')

        # 目前无持仓（没有任何标的或者该标的不在持仓标的中）但是有未成交订单：马上撤销所有
        elif (len(positions) == 0 or today_code not in [pos.contract.symbol for pos in positions]) \
                and len(open_orders) > 0:
            for od in open_orders:
                if today_code == od.contract.symbol:
                    trade_client.cancel_order(id=od.id)
                    print('cancel open order')

        # 目前无持仓且无未成交开仓订单：判断有没有一条全新的刚好走完的5min bar，有的话加入chan对象，判断是否开仓，
        # 开仓连带固定比例止损单也一起提交；不开仓则继续等；开仓则记录订单id
        elif (len(positions) == 0 or today_code not in [pos.contract.symbol for pos in positions]) \
                and len(open_orders) == 0:

            last_klu = chan[0][-1][-1]
            bsp_list = chan.get_bsp()
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]
            cur_lv_chan = chan[0]

            if cur_lv_chan[-3].idx != last_bsp.klu.klc.idx or last_bsp.klu.idx in treated_bsp_idx or \
                    not last_bsp.is_buy:
                # 已经判断过了，分型还没形成，不是买点
                continue

            last_bsp.features.add_feat(buy_stragety_feature(last_klu, cur_lv_chan, bsp_list))  # 开仓K线特征
            # 买卖点打分，应该和demo5最后的predict结果完全一致才对

            pred_prob = predict_bsp(model, last_bsp, meta)[0]
            treated_bsp_idx.add(last_bsp.klu.idx)
            if pred_prob > 0.5:
                contract = future_contract(symbol=today_code, currency='USD')
                # 生成订单对象
                buy_order = market_order(account=client_config.account, contract=contract, action='BUY', quantity=1)
                buy_oid = trade_client.place_order(buy_order)
                print(f'{cur_lv_chan[-1][-1].time}:buy id = {buy_oid}')

                # 移动止损订单
                trail_order_obj = trail_order(
                    account=client_config.account, contract=contract, action='SELL', quantity=1, trailing_percent=0.2)
                trail_oid = trade_client.place_order(trail_order_obj)
                print(f'{cur_lv_chan[-1][-1].time}:trail id = {trail_oid}')

        # 目前有持仓且无市价卖出单：监控当前价格，如果价格濒临跌破成本，撤销原有止损单，提交一个市价卖出单；否则啥也不用做，等利润奔跑；
        elif len(positions) > 0 and today_code in [pos.contract.symbol for pos in positions] and \
                'MKT' not in [od.order_type for od in open_orders]:
            sent_buy_order = max(filled_orders, key=lambda x: x.trade_time)
            now_utc = int(datetime.now(pytz.utc).timestamp())
            # 判断是否成交以及是否过去15min了
            if sent_buy_order.filled > 0 and sent_buy_order.trade_time / 1000 + 15 * 60 < now_utc:
                latest_price = quote_client.get_future_brief(identifiers=[code]).loc[0, 'latest_price']
                if latest_price < sent_buy_order.avg_fill_price + 0.5:
                    # 生成订单对象
                    contract = future_contract(symbol=sent_buy_order.contract.symbol, currency='USD')
                    # 撤销原止损单
                    trail_od_list = [od.id for od in open_orders if od.order_type == 'TRAIL']
                    if len(trail_od_list) > 0:
                        trail_oid = trail_od_list[0]
                        trade_client.cancel_order(id=trail_oid)
                    sell_order = market_order(account=client_config.account, contract=contract, action='SELL',
                                              quantity=1)
                    sell_oid = trade_client.place_order(sell_order)
                    print(f'超过15min表现不佳，止损卖出, sell id {sell_oid}')

        # 目前有该标的持仓且有市价卖出单：如果该市价卖出单超过5s则撤销掉，并发出告警，一般市价就是一定要成交的。
        elif len(positions) > 0 and today_code in [pos.contract.symbol for pos in positions] and \
                'MKT' in [od.order_type for od in open_orders]:
            for od in open_orders:
                if today_code == od.contract.symbol and od.order_type == 'MKT':
                    trade_client.cancel_order(id=od.id)
                    print('has position and mkt order not filled, cancel open order')

        # 目前有持仓却没有止损单
        elif len(positions) > 0 and today_code in [pos.contract.symbol for pos in positions] and \
                'TRAIL' not in [od.order_type for od in open_orders]:
            contract = future_contract(symbol=today_code, currency='USD')
            # 移动止损订单
            new_trail_order_obj = trail_order(
                account=client_config.account, contract=contract, action='SELL', quantity=1, trailing_percent=0.2)
            new_trail_oid = trade_client.place_order(new_trail_order_obj)
            print(f'have position but no trail order, place order trail id = {new_trail_oid}')

        # 目前有持仓，判断是否为空单，是的话赶紧平仓
        elif len(positions) > 0:
            for pos in positions:
                if today_code == pos.contract.symbol and pos.quantity < 0:
                    contract = future_contract(symbol=pos.contract.symbol, currency='USD')
                    # 生成订单对象
                    close_order = market_order(account=client_config.account, contract=contract, action='BUY',
                                               quantity=pos.quantity)
                    close_oid = trade_client.place_order(close_order)
                    print(f'why you have short position? close id = {close_oid}')

        else:
            print(f'why you can enter here?')


if __name__ == '__main__':
    # 实盘设置开盘前5分钟启动，这里end_time设置为None，会自动拉最新的1000条数据来初始化
    code = 'MRTYmain'
    # '2024-01-19 16:50:00'
    buy_model_predict(code=code, begin_time=None, end_time=None, only_bsp=True, is_send=False)
