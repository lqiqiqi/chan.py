# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import sys

sys.path.append('/root/chan.py')

from DataAPI.YFinanceAPI import YF
from get_image_api import send_msg

import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, TREND_TYPE, BSP_TYPE
from Test.config import Config


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
    first_indx = 0

    for last_5m_klu in data_src_5m.get_kl_data():
        chan.trigger_load({KL_TYPE.K_DAY: [last_5m_klu]})
        if first_indx < 30:
            first_indx += 1
            continue
        chan_snapshot = chan

        cur_lv_chan = chan_snapshot[0]
        last_last_5m_klu = [klu for ckl in cur_lv_chan[-2:] for klu in ckl][-2]
        last_last_last_5m_klu = [klu for ckl in cur_lv_chan[-3:] for klu in ckl][-3]
        four_last_5m_klu = [klu for ckl in cur_lv_chan[-4:] for klu in ckl][-4]

        if (last_5m_klu.close > last_5m_klu.trend[TREND_TYPE.MEAN][20] > last_last_5m_klu.trend[TREND_TYPE.MEAN][20] >
            last_last_last_5m_klu.trend[TREND_TYPE.MEAN][20]) and \
                (four_last_5m_klu.trend[TREND_TYPE.MEAN][20] > last_last_last_5m_klu.trend[TREND_TYPE.MEAN][20]):

            last_buy_price = last_5m_klu.close
            last_buy_time = last_5m_klu.time
            print(f'{last_buy_time}: buy price = {last_buy_price} ')
            last_break = last_5m_klu.time.to_str()

    return last_break


if __name__ == '__main__':
    res_dict = {'code': [], 'last_break': []}
    today = datetime.today()
    date_1000_days_later = today - timedelta(days=80)
    formatted_begin = date_1000_days_later.strftime("%Y-%m-%d")
    formatted_end = today.strftime("%Y-%m-%d")
    etf_df = pd.read_csv('etf_history_perfomance.csv', index_col=0).dropna()
    etf_df = etf_df[etf_df['exp_return'] > 1.5]
    for code in etf_df['code'].unique():
        try:
            last_break =\
            train_buy_model(code=code, begin_time=formatted_begin, end_time=formatted_end)
        except Exception as e:
            print(e)
            continue
        res_dict['last_break'].append(last_break)

        res_dict['code'].append(code)

    res_df = pd.DataFrame(res_dict)
    res_df['last_break'] = pd.to_datetime(res_df['last_break'])  # 将字符串转换为日期格式
    two_weeks_ago = datetime.now() - timedelta(days=3.9)  # 计算3天内日期
    res_df = res_df[res_df['last_break'] >= two_weeks_ago]  # 获取最近两周内的数据
    res_df['last_break'] = res_df['last_break'].dt.strftime('%Y-%m-%d')

    if len(res_df) > 0:
        single_stk_df = etf_df[etf_df['code'].isin(res_df['code'].unique())].set_index('code')[['exp_return', 'sector']].round(2)
        duplicated_index = single_stk_df.index.duplicated(keep='first')
        single_stk_df = single_stk_df[~duplicated_index]

        hist_perf_dict = {idx: ', '.join(f'{val}' for col, val in row.items())
                  for idx, row in single_stk_df.iterrows()}
        recent_break_dict = res_df.set_index('code')['last_break'].to_dict()
        send_msg('ma20拐头：' + str(recent_break_dict) + '\n' + str(hist_perf_dict), 'text')
    print(res_df)