# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta

from tigeropen.common.consts import SecurityType, Currency, Market

sys.path.append('/root/chan.py')

from get_image_api import send_msg
from DataAPI.YFinanceAPI import YF
from DataAPI.TigerRealAPI import TigerReal


from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Test.config import Config


def monitor(code, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    global last_last_day_klu
    print(code)
    data_src = DATA_SRC.CSV
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

    first_indx = 0
    for last_day_klu in data_src_5m.get_kl_data():
        chan.trigger_load({KL_TYPE.K_DAY: [last_day_klu]})
        chan_snapshot = chan
        if first_indx < 50:
            first_indx += 1
            continue
        cur_lv_chan = chan_snapshot[0]
        last_last_day_klu = [klu for ckl in cur_lv_chan[-2:] for klu in ckl][-2]
        last_last_last_day_klu = [klu for ckl in cur_lv_chan[-3:] for klu in ckl][-3]

    str = code
    if last_day_klu.close < last_day_klu.trend[TREND_TYPE.MEAN][20]:
        str += '\n 破线：20日均线拐头'

    if last_last_last_day_klu.trend[TREND_TYPE.MEAN][20] < last_last_day_klu.trend[TREND_TYPE.MEAN][20] and \
            last_last_day_klu.trend[TREND_TYPE.MEAN][20] > last_day_klu.trend[TREND_TYPE.MEAN][20]:
        str += '\n 拐头：20日均线拐头'

    if last_last_day_klu.trend[TREND_TYPE.MEAN][20] > last_day_klu.trend[TREND_TYPE.MEAN][20]:
        str += '\n 向下：20日均线向下'

    if last_last_day_klu.trend[TREND_TYPE.MEAN][5] > last_last_day_klu.trend[TREND_TYPE.MEAN][10] and \
            last_day_klu.trend[TREND_TYPE.MEAN][5] < last_day_klu.trend[TREND_TYPE.MEAN][10]:
        str += '\n 5日均线向下穿越10日均线'
    if str != code:
        send_msg(str, type='text')
    return


if __name__ == '__main__':
    res_dict = {'code': [], 'last_break': [], 'last_break_bsp': [], 'last_break_zs_begin_time': [],
                'last_break_zs_end_time': [], 'last_break_zs_high': []}

    data_src_instance = TigerReal("QQQ", k_type=KL_TYPE.K_5M, begin_date=None, end_date=None)
    data_src_instance.do_init()  # 初始化数据源类
    trade_client = data_src_instance.trade_client
    positions = trade_client.get_positions(sec_type=SecurityType.STK, currency=Currency.ALL, market=Market.ALL)

    today = datetime.today()
    date_120_days_later = today - timedelta(days=120)
    formatted_begin = date_120_days_later.strftime("%Y-%m-%d")
    formatted_end = today.strftime("%Y-%m-%d")

    for code in [pos.contract.identifier for pos in positions]:
    # for code in ['VNQ']:
        try:
            monitor(code=code, begin_time=formatted_begin, end_time=formatted_end)
        except Exception as e:
            send_msg(str(e), type='text')
            continue
