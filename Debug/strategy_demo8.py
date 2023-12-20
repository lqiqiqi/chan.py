#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/16 16:24
# @Author  : rockieluo
# @File    : strategy_demo8.py

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE, TREND_TYPE
from Plot.PlotDriver import CPlotDriver
from config import Config, plot_config, plot_para

if __name__ == "__main__":
    """
    (废弃，没有调试成功)本方案尝试区间套
    """
    # code = "sz.000001"
    code = 'HK.00700'
    # code = 'HK.09868'
    code_list = ['']
    begin_time = "2023-01-01"
    end_time = "2023-10-31"
    # data_src = DATA_SRC.BAO_STOCK
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]

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

    is_hold = False
    last_buy_price = None
    used_bsp_list = []
    for chan_snapshot in chan.step_load():  # 每增加一根K线，返回当前静态精算结果
        bsp_list = chan_snapshot[-1].seg_bs_point_lst.getLastestBspList()  # 获取5min买卖点列表
        if not bsp_list:  # 为空
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[-1]
        last_klu = chan_snapshot[-1][-1][-1]

        if (BSP_TYPE.T2 in last_bsp.type) and last_klu.sup_kl.close > last_klu.sup_kl.trend[TREND_TYPE.MEAN][60]\
                and last_bsp.is_buy is True:
            # cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and
            if last_bsp.is_buy and not is_hold and last_bsp.klu.time not in used_bsp_list:
                plot_driver = CPlotDriver(
                    chan_snapshot,
                    plot_config=plot_config,
                    plot_para=plot_para,
                )
                plot_driver.figure.show()
                last_buy_price = cur_lv_chan[-1][-1].close  # 开仓价格为最后一根K线close
                print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
                used_bsp_list.append(last_bsp.klu.time)
                is_hold = True

        # 重置bsp
        last_bsp = bsp_list[-1]
        if is_hold:
            if (BSP_TYPE.T2 in last_bsp.type and last_bsp.is_buy is False) or \
                    (BSP_TYPE.T2S in last_bsp.type and last_bsp.is_buy is False) or \
                    last_klu.sup_kl.close < last_klu.sup_kl.trend[TREND_TYPE.MEAN][60]:
                sell_price = cur_lv_chan[-1][-1].close
                plot_driver = CPlotDriver(
                    chan_snapshot,
                    plot_config=plot_config,
                    plot_para=plot_para,
                )
                plot_driver.figure.show()
                print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {(sell_price-last_buy_price)/last_buy_price*100:.2f}%')
                is_hold = False
