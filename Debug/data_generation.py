import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from Plot.PlotDriver import CPlotDriver

if __name__ == "__main__":
    """
    """
    # code = "sz.000001"
    # code = 'HK.00700'
    code = 'HK.09868'
    begin_time = "2018-01-01"
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
        "one_bi_zs": False,
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

    data_dict = {
        'close_price': [],
        'macd': [],
        'direct': [],
        'bi_dir': [],
        'zs_high': [],
        'zs_low': [],
        'zs_begin_idx': [],
        'zs_end_idx': [],
        'is_buy': [],
        'one_bsp': [],
        'two_bsp': [],
        'two_s_bsp': [],
        'three_bsp': [],
        'last_bsp_idx': []
    }
    for chan_snapshot in chan.step_load():  # 每增加一根K线，返回当前静态精算结果

        cur_lv_chan = chan_snapshot[0]

        # k线粒度
        cur_kline = cur_lv_chan[-1][-1]
        klu_idx = cur_kline.idx
        cur_date = cur_kline.time.toDateStr('-')
        if cur_date < begin_time:
            continue
        close_price = cur_kline.close
        macd = cur_kline.macd.macd

        # 合并后k线粒度
        cur_ckline = cur_lv_chan[-1]
        idx = cur_ckline.idx
        if FX_TYPE.BOTTOM == cur_ckline.fx:
            direct = 1
        elif FX_TYPE.TOP == cur_ckline.fx:
            direct = -1
        else:
            direct = 0

        if len(cur_lv_chan.bi_list) > 0:
            bi_dir = 1 if cur_lv_chan.bi_list[-1].is_up is True else 0
        else:
            bi_dir = 0

        # 中枢
        if len(cur_lv_chan.zs_list) > 0:
            zs_high = (close_price - cur_lv_chan.zs_list[-1].high) / cur_lv_chan.zs_list[-1].high
            zs_low = (close_price - cur_lv_chan.zs_list[-1].low) / cur_lv_chan.zs_list[-1].low
            zs_begin_idx = idx - cur_lv_chan.zs_list[-1].begin_bi.begin_klc.idx
            zs_end_idx = idx - cur_lv_chan.zs_list[-1].end_bi.end_klc.idx
        else:
            zs_high = 0
            zs_low = 0
            zs_begin_idx = 0
            zs_end_idx = 0

        # bsp
        bsp_list = chan_snapshot.get_bsp()  # 获取买卖点列表
        if len(bsp_list) > 0:
            last_bsp = bsp_list[-1]
            is_buy = 1 if last_bsp.is_buy is True else -1
            one_bsp = 1 if BSP_TYPE.T1P in last_bsp.type or BSP_TYPE.T1 in last_bsp.type else -1
            two_bsp = 1 if BSP_TYPE.T2 in last_bsp.type else -1
            two_s_bsp = 1 if BSP_TYPE.T2S in last_bsp.type else -1
            three_bsp = 1 if BSP_TYPE.T3B in last_bsp.type or BSP_TYPE.T3A in last_bsp.type else -1
            last_bsp_idx = klu_idx - last_bsp.klu.idx
        else:
            is_buy = 0
            one_bsp = 0
            two_bsp = 0
            two_s_bsp = 0
            three_bsp = 0
            last_bsp_idx = 0

        data_dict['macd'].append(macd)
        data_dict['direct'].append(direct)
        data_dict['bi_dir'].append(bi_dir)
        data_dict['zs_high'].append(zs_high)
        data_dict['zs_low'].append(zs_low)
        data_dict['zs_begin_idx'].append(zs_begin_idx)
        data_dict['zs_end_idx'].append(zs_end_idx)
        data_dict['is_buy'].append(is_buy)
        data_dict['one_bsp'].append(one_bsp)
        data_dict['two_bsp'].append(two_bsp)
        data_dict['two_s_bsp'].append(two_s_bsp)
        data_dict['three_bsp'].append(three_bsp)
        data_dict['close_price'].append(close_price)
        data_dict['last_bsp_idx'].append(last_bsp_idx)

    df = pd.DataFrame(data_dict)
    df['close_shift'] = df['close_price'].shift(-7)
    df['return_rate'] = (df['close_shift'] - df['close_price'])/df['close_price']*100
    df['return_rate'] = df['return_rate'].apply(lambda x: 1 if x > 0 else 0)
    df = df[df['is_buy'] != 0]
    df = df.dropna()
    df = df.drop(['close_shift', 'close_price'], axis=1)
    # print(df)
    df.to_csv('../Data/09868.csv')








