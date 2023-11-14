from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from DataAPI.BaoStockAPI import CBaoStock
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

import baostock as bs
import baostock as bs

if __name__ == "__main__":
    # code_dict = {
    #     # "sh.000991": "医药etf",
    #     # "sh.600185": "格立地产",
    #     # "sz.002835": "同为股份",
    #     # "sh.605118": "力鼎光电",
    #     # "sz.002765": "蓝带科技",
    #     # "sh.603316": "诚邦股份",
    #     "sz.300418": "昆仑万维",
    # }

    lg = bs.login()
    # 获取沪深300成分股
    rs = bs.query_hs300_stocks()
    # 登出系统
    bs.logout()

    code_dict = {}
    for stock_item in rs.data:
        code = stock_item[1]
        name = stock_item[2]
        code_dict[code] = name

    begin_time = "2023-03-01"
    # end_time = None
    end_time = "2023-11-06"
    data_src = DATA_SRC.BAO_STOCK
    # data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
        "bi_strict": False,
        "bi_fx_check": "loss",
        "bi_algo": "normal",
        "bi_end_is_peak": False,
        "one_bi_zs": True,
        "triger_step": False,
        "skip_step": 0,
        # "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,1p,3a,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
        # "divergence_rate": 200
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
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
            # "sub_lv_cnt": 6,
            # "facecolor": 'green',
            "plot_trendline": False
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
            "sub_lv_cnt": 6,
            "facecolor": 'green'
        },
        "figure": {
            "x_range": 200,
            "only_top_lv": False
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }
    for code, v in code_dict.items():
        chan = CChan(
            code=code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=lv_list,
            config=config,
            autype=AUTYPE.QFQ,
        )
        bsp_list = chan.get_bsp(idx=0)
        if bsp_list:
            for bsp in bsp_list[-2:]:
                if (BSP_TYPE.T3B in bsp.type or BSP_TYPE.T3A in bsp.type) and bsp.is_buy:
                    # or BSP_TYPE.T2S in bsp.type
                    print(code, v, bsp.klu.time.toDateStr(), bsp.type2str())
                    if not config.triger_step:
                        plot_driver = CPlotDriver(
                            chan,
                            plot_config=plot_config,
                            plot_para=plot_para,
                        )
                        plot_driver.figure.show()
