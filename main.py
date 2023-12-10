from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver
from Test.config import plot_config, plot_para, Config

if __name__ == "__main__":
    code = "SPY"
    begin_time = "2023-12-03"
    end_time = "2023-12-10"
    data_src = DATA_SRC.YFINANCE
    lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_1M]

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
    # bsp_list = chan.get_seg_bsp(idx=0)

    if not config.triger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        plot_driver.figure.show()
    else:
        CAnimateDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
