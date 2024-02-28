# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta

sys.path.append('/root/chan.py')

from get_image_api import send_msg
from DataAPI.YFinanceAPI import YF

import numpy as np
import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Test.config import Config


def cal_atr(cur_lv_chan, n=14):
    high = np.array([klu.high for ckl in cur_lv_chan[-100:] for klu in ckl])
    low = np.array([klu.low for ckl in cur_lv_chan[-100:] for klu in ckl])
    close = np.array([klu.close for ckl in cur_lv_chan[-100:] for klu in ckl])
    atr_sum = 0
    for i in range(n, 0, -1):
        tr = max(high[-i] - low[-i], abs(high[-i] - close[-(i + 1)]), abs(low[-i] - close[-(i + 1)]))
        atr_sum += tr

    atr = atr_sum / n
    return atr


def train_buy_model(code, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    print(code)
    data_src = DATA_SRC.CSV
    # data_src = DATA_SRC.TigerMock
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

    data_src_5m = YF(code, k_type=KL_TYPE.K_DAY, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)

    last_break = None
    last_break_bsp = None
    last_break_zs_begin_time = None
    last_break_zs_end_time = None
    last_break_zs_high = None
    first_indx = 0

    for last_5m_klu in data_src_5m.get_kl_data():
        chan.trigger_load({KL_TYPE.K_DAY: [last_5m_klu]})
        if first_indx < 365:
            first_indx += 1
            continue
        chan_snapshot = chan
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        bp_list = [bsp for bsp in bsp_list if bsp.is_buy]
        if not bp_list:
            continue
        last_bp = bp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        last_zs = cur_lv_chan.zs_list[-1]
        last_zs_high = max([ckl.high for bi in last_zs.bi_lst for ckl in bi.klc_lst])
        last_last_5m_klu = [klu for ckl in cur_lv_chan[-2:] for klu in ckl][-2]
        last_last_last_5m_klu = [klu for ckl in cur_lv_chan[-3:] for klu in ckl][-3]
        atr = cal_atr(cur_lv_chan)
        last_bi = cur_lv_chan.bi_list[-1]
        last_bi_highest_close = max([klu.close for klc in last_bi.klc_lst for klu in klc])
        if len(cur_lv_chan.seg_list) < 2:
            return None
        last_last_seg = cur_lv_chan.seg_list[-2]

        if (last_last_5m_klu.close < last_zs_high + atr < last_5m_klu.close or
            last_last_last_5m_klu.close < last_zs_high + atr < last_last_5m_klu.close) \
                and not (BSP_TYPE.T1P in last_bp.type or BSP_TYPE.T1 in last_bp.type) and \
                last_5m_klu.close >= last_bi_highest_close:
            if last_zs.begin_bi.get_begin_klu().time < last_last_seg.get_end_klu().time and not last_last_seg.is_up():
                continue
            bi_distance = last_bi.idx - last_zs.end_bi.idx
            if bi_distance > 2:
                continue

            last_buy_price = last_5m_klu.close
            last_buy_time = last_5m_klu.time
            print(f'{last_buy_time}: buy price = {last_buy_price} ')

            last_break = last_5m_klu.time.to_str()
            last_break_bsp = last_bp.klu.time
            last_break_zs_begin_time = last_zs.begin.time
            last_break_zs_end_time = last_zs.end.time
            last_break_zs_high = last_zs_high

    return last_break, last_break_bsp, last_break_zs_begin_time, last_break_zs_end_time, last_break_zs_high


if __name__ == '__main__':
    res_dict = {'code': [], 'last_break': [], 'last_break_bsp': [], 'last_break_zs_begin_time': [],
                'last_break_zs_end_time': [], 'last_break_zs_high': []}

    today = datetime.today()
    date_1000_days_later = today - timedelta(days=1200)
    formatted_begin = date_1000_days_later.strftime("%Y-%m-%d")
    formatted_end = today.strftime("%Y-%m-%d")

    for code in ['VTI', 'OEF', 'SPY', 'DIA', 'MDY', 'RSP', 'QQQ', 'QTEC', 'IWB', 'IWM', 'MTUM', 'SPHB',
                 'QUAL', 'SPLV', 'RSPC', 'RSPD', 'RSPS', 'RSPG', 'RSPF', 'RSPH', 'RSPN', 'RSPM', 'RSPR', 'RSPT',
                 'RSPU', 'IWY', 'IVW', 'IWF', 'IWO', 'METV', 'IPO', 'SNSR', 'XT', 'MOAT', 'SOCL', 'ONLN', 'SKYY',
                 'HERO', 'IBUY', 'IPAY', 'FINX', 'CIBR', 'IGF', 'DRIV', 'BOTZ', 'ROBO', 'MOO', 'TAN', 'QCLN', 'PBW']:

        try:
            last_break, last_break_bsp, last_break_zs_begin_time, last_break_zs_end_time, last_break_zs_high = \
                train_buy_model(code=code, begin_time=formatted_begin, end_time=formatted_end)
        except Exception as e:
            print(e)
            continue
        res_dict['last_break'].append(last_break)
        res_dict['last_break_bsp'].append(last_break_bsp)
        res_dict['last_break_zs_begin_time'].append(last_break_zs_begin_time)
        res_dict['last_break_zs_end_time'].append(last_break_zs_end_time)
        res_dict['last_break_zs_high'].append(last_break_zs_high)
        res_dict['code'].append(code)
    res_df = pd.DataFrame(res_dict)
    res_df['last_break'] = pd.to_datetime(res_df['last_break'])  # 将字符串转换为日期格式
    two_weeks_ago = datetime.now() - timedelta(days=3)  # 计算3天前日期
    res_df = res_df[res_df['last_break'] > two_weeks_ago]  # 获取最近两周内的数据
    res_df['last_break'] = res_df['last_break'].dt.strftime('%Y-%m-%d')

    if len(res_df) > 0:
        recent_break_dict = res_df.set_index('code')['last_break'].to_dict()
        send_msg('近3天突破中枢高点：' + str(recent_break_dict), 'text')
    print(res_df)
