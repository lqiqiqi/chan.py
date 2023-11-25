#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/19 11:26
# @Author  : rockieluo
# @File    : check_is_1_3_bp.py

from datetime import timedelta, datetime

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE, FX_TYPE

from Test.config import Config, plot_config, plot_para


def check_top_fx(code, date):
    date_format = "%Y-%m-%d"
    date_object = datetime.strptime(date, date_format)
    begin_date = (date_object - timedelta(days=7)).strftime('%Y-%m-%d')

    begin_time = begin_date
    end_time = date
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

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

    last_fx = next((item.fx for item in reversed(chan.kl_datas[KL_TYPE.K_DAY].lst) if item.fx != FX_TYPE.UNKNOWN),
                   FX_TYPE.TOP)
    if last_fx == FX_TYPE.TOP:
        return {'result': True, 'message': 'sell because of top fx'}
    else:
        return {'result': False, 'message': 'no top fx'}


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

    bsp1_list = chan.get_seg_bsp(idx=0)
    if len(bsp1_list) > 0:
        bsp1 = bsp1_list[-1]
        last_seg = chan.kl_datas[KL_TYPE.K_30M].seg_list[-1]
        last_bi = chan.kl_datas[KL_TYPE.K_30M].bi_list[-1]
        last_fx = next((item.fx for item in reversed(chan.kl_datas[KL_TYPE.K_30M].lst) if item.fx != FX_TYPE.UNKNOWN),
                       FX_TYPE.TOP)
        if ((BSP_TYPE.T1 in bsp1.type) or (BSP_TYPE.T1P in bsp1.type)) and bsp1.is_buy \
                and last_fx != FX_TYPE.TOP and (last_bi.is_up()):
            return {'result': True, 'message': '30min 1类买点'}
        if len(bsp1_list) > 1:
            last_bsp1 = bsp1_list[-2]
            if ((BSP_TYPE.T2 in bsp1.type) and (BSP_TYPE.T1 in last_bsp1.type)) and last_bsp1.is_buy and bsp1.is_buy\
                    and last_bsp1.klu.close < bsp1.klu.close and last_fx != FX_TYPE.TOP  and (last_bi.is_up()):
                return {'result': True, 'message': '30min 2类买点且高于1类'}
        segzs_list = chan.kl_datas[KL_TYPE.K_30M].segzs_list
        if len(segzs_list) > 0:
            last_zs = segzs_list.zs_lst[-1]
            if last_zs.bi_out == last_seg and last_zs.high < chan.kl_datas[KL_TYPE.K_30M].lst[-1].low and \
                    last_fx != FX_TYPE.TOP  and (last_bi.is_up()):
                return {'result': True, 'message': '30min 突破中枢'}
        # if res['result']:
        #     from Plot.PlotDriver import CPlotDriver
        #     plot_driver = CPlotDriver(
        #         chan,
        #         plot_config=plot_config,
        #         plot_para=plot_para,
        #     )
        #     plot_driver.figure.show()
        #     return res

    bsp2_list = chan.get_seg_bsp(idx=1)
    if len(bsp2_list) > 0:
        bsp2 = bsp2_list[-1]
        last_seg = chan.kl_datas[KL_TYPE.K_5M].seg_list[-1]
        last_bi = chan.kl_datas[KL_TYPE.K_30M].bi_list[-1]
        last_fx = next((item.fx for item in reversed(chan.kl_datas[KL_TYPE.K_5M].lst) if item.fx != FX_TYPE.UNKNOWN),
                       FX_TYPE.TOP)
        if ((BSP_TYPE.T1 in bsp2.type) or (BSP_TYPE.T1P in bsp2.type)) and bsp2.is_buy \
                and last_fx != FX_TYPE.TOP and (last_bi.is_up()):
            return {'result': True, 'message': '5min 1类买点'}
        if len(bsp2_list) > 1:
            last_bsp2 = bsp2_list[-2]
            if ((BSP_TYPE.T2 in bsp2.type) and (BSP_TYPE.T1 in last_bsp2.type)) and last_bsp2.is_buy and bsp2.is_buy\
                    and last_bsp2.klu.close < bsp2.klu.close and last_fx != FX_TYPE.TOP and (last_bi.is_up()):
                return {'result': True, 'message': '5min 2类买点且高于1类'}
        segzs_list = chan.kl_datas[KL_TYPE.K_5M].segzs_list
        if len(segzs_list) > 0:
            last_zs = segzs_list.zs_lst[-1]
            if last_zs.bi_out == last_seg and last_zs.high < chan.kl_datas[KL_TYPE.K_5M].lst[-1].low and \
                    last_fx != FX_TYPE.TOP and (last_bi.is_up()):
                return {'result': True, 'message': '5min 突破中枢'}
        # if res['result']:
        #     from Plot.PlotDriver import CPlotDriver
        #     plot_driver = CPlotDriver(
        #         chan,
        #         plot_config=plot_config,
        #         plot_para=plot_para,
        #     )
        #     plot_driver.figure.show()
        #     return res

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

    code = "sh.600319"
    end_time = "2023-04-09"
    check_is_1_3_bsp(code, end_time)

    # stock_list = ['sz.002899',
    #              'sh.600202',
    #              'sz.300387',
    #              'sz.300970',
    #              'sz.002857',
    #              'sz.000691',
    #              'sh.600250',
    #              'sz.300923',
    #              'sh.603022',
    #              'sh.603488',
    #              'sz.301167',
    #              'sz.300717',
    #              'sz.300164',
    #              'sz.300371',
    #              'sz.002862',
    #              'sh.603089',
    #              'sz.301007',
    #              'sz.003003',
    #              'sz.301065',
    #              'sz.301186',
    #              'sz.300923',
    #              'sz.002890',
    #              'sh.603159',
    #              'sz.002826',
    #              'sh.600051',
    #              'sz.301007',
    #              'sz.300886',
    #              'sz.002188',
    #              'sz.000608',
    #              'sh.603767',
    #              'sz.301053',
    #              'sz.300923',
    #              'sh.600319',
    #              'sh.600051',
    #              'sz.300851',
    #              'sz.301119',
    #              'sz.300635',
    #              'sh.603321',
    #              'sz.300886',
    #              'sz.000632',
    #              'sz.002890',
    #              'sz.300923',
    #              'sh.603159',
    #              'sz.000691',
    #              'sz.300931',
    #              'sh.600319',
    #              'sz.300417',
    #              'sz.300549',
    #              'sh.600083',
    #              'sh.603321',
    #              'sz.300923',
    #              'sz.003003',
    #              'sz.301008',
    #              'sz.000691',
    #              'sz.003023',
    #              'sz.002998',
    #              'sh.603499',
    #              'sh.603022',
    #              'sz.300749',
    #              'sz.002898',
    #              'sz.000691',
    #              'sz.300819',
    #              'sh.600272',
    #              'sz.301010',
    #              'sz.300971',
    #              'sz.002144',
    #              'sh.600857',
    #              'sh.600689',
    #              'sz.002778',
    #              'sz.002899',
    #              'sh.603499',
    #              'sz.002778',
    #              'sz.003008',
    #              'sz.000605',
    #              'sz.003023',
    #              'sz.003017',
    #              'sz.300971',
    #              'sz.300948',
    #              'sh.603161',
    #              'sz.301098',
    #              'sz.002778',
    #              'sz.301163',
    #              'sh.600793',
    #              'sh.603657',
    #              'sz.300899',
    #              'sz.002767',
    #              'sz.300509',
    #              'sz.002591',
    #              'sz.002188',
    #              'sz.000622',
    #              'sz.000632',
    #              'sz.003017',
    #              'sz.003003',
    #              'sh.600235',
    #              'sh.603729',
    #              'sz.300877',
    #              'sh.603499',
    #              'sh.600137',
    #              'sz.301125',
    #              'sh.603037',
    #              'sz.300126',
    #              'sz.002591',
    #              'sh.603729',
    #              'sz.002780',
    #              'sz.301156',
    #              'sz.000632',
    #              'sz.300535',
    #              'sz.301233',
    #              'sz.300819',
    #              'sz.002278',
    #              'sh.603090',
    #              'sz.300877',
    #              'sh.603221',
    #              'sh.600202',
    #              'sz.002820',
    #              'sh.603696']
    #
    # date_list = ['2023-01-08',
    #              '2023-01-08',
    #              '2023-01-08',
    #              '2023-01-08',
    #              '2023-02-05',
    #              '2023-01-15',
    #              '2023-02-05',
    #              '2023-01-02',
    #              '2023-01-02',
    #              '2023-02-12',
    #              '2023-01-08',
    #              '2023-02-12',
    #              '2023-02-05',
    #              '2023-02-12',
    #              '2023-02-26',
    #              '2023-02-26',
    #              '2023-01-02',
    #              '2023-02-19',
    #              '2023-03-05',
    #              '2023-03-05',
    #              '2023-01-02',
    #              '2023-01-15',
    #              '2023-01-29',
    #              '2023-03-12',
    #              '2023-03-12',
    #              '2023-01-02',
    #              '2023-01-02',
    #              '2023-02-19',
    #              '2023-03-19',
    #              '2023-03-05',
    #              '2023-01-02',
    #              '2023-01-02',
    #              '2023-03-26',
    #              '2023-03-12',
    #              '2023-01-29',
    #              '2023-03-26',
    #              '2023-01-08',
    #              '2023-03-26',
    #              '2023-01-02',
    #              '2023-04-02',
    #              '2023-01-15',
    #              '2023-01-02',
    #              '2023-01-29',
    #              '2023-01-15',
    #              '2023-01-02',
    #              '2023-03-26',
    #              '2023-04-16',
    #              '2023-01-02',
    #              '2023-01-08',
    #              '2023-03-26',
    #              '2023-01-02',
    #              '2023-02-19',
    #              '2023-05-03',
    #              '2023-01-15',
    #              '2023-05-14',
    #              '2023-04-23',
    #              '2023-06-04',
    #              '2023-01-02',
    #              '2023-05-03',
    #              '2023-05-14',
    #              '2023-01-15',
    #              '2023-05-07',
    #              '2023-06-11',
    #              '2023-06-18',
    #              '2023-06-18',
    #              '2023-07-02',
    #              '2023-06-11',
    #              '2023-05-03',
    #              '2023-05-07',
    #              '2023-01-08',
    #              '2023-06-04',
    #              '2023-05-07',
    #              '2023-01-08',
    #              '2023-07-30',
    #              '2023-05-14',
    #              '2023-07-30',
    #              '2023-06-18',
    #              '2023-05-28',
    #              '2023-08-06',
    #              '2023-08-06',
    #              '2023-05-07',
    #              '2023-07-30',
    #              '2023-05-21',
    #              '2023-05-21',
    #              '2023-07-02',
    #              '2023-02-19',
    #              '2023-08-27',
    #              '2023-08-27',
    #              '2023-02-19',
    #              '2023-08-27',
    #              '2023-04-02',
    #              '2023-07-30',
    #              '2023-02-19',
    #              '2023-09-03',
    #              '2023-09-03',
    #              '2023-09-03',
    #              '2023-06-04',
    #              '2023-09-03',
    #              '2023-09-03',
    #              '2023-09-10',
    #              '2023-09-10',
    #              '2023-08-27',
    #              '2023-09-03',
    #              '2023-09-24',
    #              '2023-09-24',
    #              '2023-04-02',
    #              '2023-10-08',
    #              '2023-10-08',
    #              '2023-05-07',
    #              '2023-02-19',
    #              '2023-10-15',
    #              '2023-09-03',
    #              '2023-10-15',
    #              '2023-01-08',
    #              '2023-10-08',
    #              '2023-06-11']
    #
    # for stock, date in zip(stock_list, date_list):
    #     res = check_is_1_3_bsp_test(stock, date)
    #     print(stock, date, res)

