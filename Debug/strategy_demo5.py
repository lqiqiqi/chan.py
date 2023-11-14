from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from Plot.PlotDriver import CPlotDriver

if __name__ == "__main__":
    """
    本方案在三类买点出现时建仓，且前面的中枢足够打，在一类macd死叉时平仓，效果不佳
    """
    # code = "sz.000001"
    code = 'HK.00700'
    # code = 'HK.09868'
    code_list = ['']
    begin_time = "2023-01-01"
    end_time = "2023-10-31"
    # data_src = DATA_SRC.BAO_STOCK
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
        "triger_step": True,  # 打开开关！
        "bi_strict": False,
        "bi_fx_check": "loss",
        "bi_algo": "normal",
        "bi_end_is_peak": False,
        "one_bi_zs": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "max_bs2_rate": 0.618,
        "bs_type": '1,2,1p,3a,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })
    # {
    #     "triger_step": True,  # 打开开关！
    #     "divergence_rate": 0.8,
    #     "min_zs_cnt": 1,
    # }
    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": True,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
            "sub_lv_cnt": None
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
            # "sub_lv_cnt": None
        },
        "figure": {
            "x_range": 400,
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }


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
        bsp_list = chan_snapshot.get_bsp()  # 获取买卖点列表
        if not bsp_list:  # 为空
            continue
        last_bsp = bsp_list[-1]
        if BSP_TYPE.T1P in last_bsp.type and len(bsp_list) >= 2:
            # 盘整1类卖点可能会和3类买点同时出现，此时不能取最后一个bsp，要取倒数第二个，同时后面要重置last_bsp
            last_bsp = bsp_list[-2]

        cur_lv_chan = chan_snapshot[0]
        if (BSP_TYPE.T3A in last_bsp.type or BSP_TYPE.T3B in last_bsp.type) and \
            (cur_lv_chan.zs_list[-2].end_bi.idx - cur_lv_chan.zs_list[-2].begin_bi.idx > 3 or
                    (cur_lv_chan.zs_list[-2].low < cur_lv_chan.zs_list[-3].high or
                     cur_lv_chan.zs_list[-2].high < cur_lv_chan.zs_list[-3].low)):  # 加入只做3类买卖点
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
            if cur_lv_chan[-3][-1].macd.DIF > cur_lv_chan[-3][-1].macd.DEA \
                    and cur_lv_chan[-2][-1].macd.DIF < cur_lv_chan[-2][-1].macd.DEA \
                    and cur_lv_chan[-1][-1].macd.DIF < cur_lv_chan[-1][-1].macd.DEA:
                sell_price = cur_lv_chan[-1][-1].close
                plot_driver = CPlotDriver(
                    chan_snapshot,
                    plot_config=plot_config,
                    plot_para=plot_para,
                )
                plot_driver.figure.show()
                print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {(sell_price-last_buy_price)/last_buy_price*100:.2f}%')
                is_hold = False
