from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE, TREND_TYPE
from DataAPI.BaoStockAPI import CBaoStock
from DataAPI.FuTuAPI import Futu
from Plot.PlotDriver import CPlotDriver
from config import Config, plot_config, plot_para

if __name__ == "__main__":
    """
    本方案尝试5min线段买卖点，30min上60ma
    
    本demo演示当你要用多级别trigger load的时候，如何巧妙的解决时间对齐的问题：
    解决方案就是喂第一根最大级别的时候，直接把所有的次级别全部喂进去
    之后每次只需要喂一根最大级别即可，复用框架内置的K线时间对齐能力
    从而避免了在trigger_load入参那里进行K线对齐，自己去找出最大级别那根K线下所有次级别的K线
    """
    code = 'HK.00700'
    begin_time = "2023-01-01"
    end_time = "2023-10-31"
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]

    config_object = Config()
    chan_config = config_object.read_chan_config_trigger_step
    config = CChanConfig(chan_config)

    chan = CChan(
        code=code,
        begin_time=begin_time,  # 已经没啥用了这一行
        end_time=end_time,  # 已经没啥用了这一行
        data_src=data_src,  # 已经没啥用了这一行
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,  # 已经没啥用了这一行
    )
    Futu.do_init()
    data_src_30m = Futu(code, k_type=KL_TYPE.K_30M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    data_src_5m = Futu(code, k_type=KL_TYPE.K_5M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_5m_all = list(data_src_5m.get_kl_data())

    used_bsp_list = []
    is_hold = False
    for _idx, klu in enumerate(data_src_30m.get_kl_data()):
        # 本质是每喂一根日线的时候，这根日线之前的都要喂过，提前喂多点不要紧，框架会自动根据日线来截取需要的30M K线
        # 5M一口气全部喂完，后续就不用关注时间对齐的问题了
        if _idx == 0:
            chan.trigger_load({KL_TYPE.K_30M: [klu], KL_TYPE.K_5M: kl_5m_all})
        else:
            chan.trigger_load({KL_TYPE.K_30M: [klu]})

        # 检查时间对齐
        print("当前所有30min线:", [klu.time.to_str() for klu in chan[0].klu_iter()][-1])
        # print("当前所有5min线:", [klu.time.to_str() for klu in chan[1].klu_iter()][-6:], "\n")

        last_30min = [klu for klu in chan[0].klu_iter()][-1]
        last_5min = [klu for klu in chan[1].klu_iter()][-1]
        bsp_list = chan.kl_datas[KL_TYPE.K_5M].seg_bs_point_lst.getLastestBspList()  # 获取5min买卖点列表
        if not len(bsp_list) > 1:  # 为空
            continue
        last_bsp = bsp_list[0]
        last_last_bsp = bsp_list[1]
        if (BSP_TYPE.T2 in last_bsp.type) and \
                (BSP_TYPE.T1 in last_last_bsp.type or BSP_TYPE.T1P in last_last_bsp.type) and \
                last_30min.close > last_30min.trend[TREND_TYPE.MEAN][60] and \
                last_bsp.is_buy is True and last_last_bsp.is_buy is True and\
                not chan[1].bi_list[-1].is_down():
            # cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and
            if last_bsp.is_buy and not is_hold and last_bsp.klu.time not in used_bsp_list:
                plot_driver = CPlotDriver(
                    chan,
                    plot_config=plot_config,
                    plot_para=plot_para,
                )
                plot_driver.figure.show()
                last_buy_price = last_5min.close  # 开仓价格为最后一根K线close
                print(f'{last_5min.time}:buy price = {last_buy_price}')
                used_bsp_list.append(last_bsp.klu.time)
                is_hold = True

        # # 重置bsp
        # last_bsp = bsp_list[-1]
        if is_hold:
            if (
                    (BSP_TYPE.T2 in last_bsp.type and last_bsp.is_buy is False) or \
                    (BSP_TYPE.T2S in last_bsp.type and last_bsp.is_buy is False) or \
                    (last_30min.close < last_buy_price*0.95)
                ) \
                    and \
                    chan[1].bi_list[-1].is_down():
                sell_price = last_5min.close
                plot_driver = CPlotDriver(
                    chan,
                    plot_config=plot_config,
                    plot_para=plot_para,
                )
                plot_driver.figure.show()
                print(f'{last_5min.time}:sell price = {sell_price}, profit rate = '
                      f'{(sell_price-last_buy_price)/last_buy_price*100:.2f}%')
                is_hold = False

    Futu.do_close()
