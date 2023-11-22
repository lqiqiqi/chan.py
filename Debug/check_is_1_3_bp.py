#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 11:26
# @Author  : rockieluo
# @File    : check_is_1_3_bp.py

from datetime import timedelta, datetime

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE

from Test.config import Config, plot_config, plot_para


def check_is_1_3_bsp(code, date):

    date_format = "%Y-%m-%d"
    date_object = datetime.strptime(date, date_format)
    begin_date = (date_object - timedelta(days=180)).strftime('%Y-%m-%d')

    begin_time = begin_date
    end_time = date
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]

    config_object = Config()
    chan_config = config_object.read_chan_config
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

    # from Plot.PlotDriver import CPlotDriver
    # plot_driver = CPlotDriver(
    #     chan,
    #     plot_config=plot_config,
    #     plot_para=plot_para,
    # )
    # plot_driver.figure.show()

    bsp1_list = chan.get_seg_bsp(idx=0)
    if len(bsp1_list) > 0:
        bsp1 = bsp1_list[-1]
        last_seg = chan.kl_datas[KL_TYPE.K_30M].seg_list[-1]
        last_bi = chan.kl_datas[KL_TYPE.K_30M].bi_list[-1]
        if ((BSP_TYPE.T1 in bsp1.type) or (BSP_TYPE.T1P in bsp1.type)) and bsp1.is_buy \
                and (last_seg.is_up() or last_bi.is_up()):
            return {'result': True, 'message': '30min 1类买点'}
        if len(bsp1_list) > 1:
            last_bsp1 = bsp1_list[-2]
            if ((BSP_TYPE.T2 in bsp1.type) and (BSP_TYPE.T1 in last_bsp1.type)) and last_bsp1.klu.close < bsp1.klu.close and (last_seg.is_up() or last_bi.is_up()):
                return {'result': True, 'message': '30min 2类买点且高于1类'}
        segzs_list = chan.kl_datas[KL_TYPE.K_30M].segzs_list
        if len(segzs_list) > 0:
            last_zs = segzs_list.zs_lst[-1]
            if last_zs.bi_out == last_seg and last_zs.high < chan.kl_datas[KL_TYPE.K_30M].lst[-1].low and \
                    (last_seg.is_up() or last_bi.is_up()):
                return {'result': True, 'message': '30min 突破中枢'}

    bsp2_list = chan.get_seg_bsp(idx=1)
    if len(bsp2_list) > 0:
        bsp2 = bsp2_list[-1]
        last_seg = chan.kl_datas[KL_TYPE.K_5M].seg_list[-1]
        last_bi = chan.kl_datas[KL_TYPE.K_30M].bi_list[-1]
        if ((BSP_TYPE.T1 in bsp2.type) or (BSP_TYPE.T1P in bsp2.type)) and bsp2.is_buy and last_seg.is_up()\
                and (last_seg.is_up() or last_bi.is_up()):
            return {'result': True, 'message': '5min 1类买点'}
        if len(bsp2_list) > 1:
            last_bsp2 = bsp2_list[-2]
            if ((BSP_TYPE.T2 in bsp2.type) and (BSP_TYPE.T1 in last_bsp2.type))\
                    and last_bsp2.klu.close < bsp2.klu.close and (last_seg.is_up() or last_bi.is_up()):
                return {'result': True, 'message': '5min 2类买点且高于1类'}
        segzs_list = chan.kl_datas[KL_TYPE.K_5M].segzs_list
        if len(segzs_list) > 0:
            last_zs = segzs_list.zs_lst[-1]
            if last_zs.bi_out == last_seg and last_zs.high < chan.kl_datas[KL_TYPE.K_5M].lst[-1].low and \
                    (last_seg.is_up() or last_bi.is_up()):
                return {'result': True, 'message': '5min 突破中枢'}

    return {'result': False, 'message': '无有用信号'}


if __name__ == '__main__':
    # 测试一类买点
    code = "sz.002596"
    end_time = "2023-10-30"

    # 测试二类买点且高于一类
    code = "sz.002343"
    end_time = "2023-10-30"

    # 测试突破中枢
    code = "sh.603496"
    end_time = "2023-09-27"

    res = check_is_1_3_bsp(code, end_time)
    print(res)

