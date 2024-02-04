import os

import datetime
import time
import unittest
import pandas as pd
import pytz

from tigeropen.common.consts import BarPeriod
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config, TigerOpenClientConfig

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# client_config = get_client_config(private_key_path='your private key path',
#                                   tiger_id='your tiger id',
#                                   account='your account'
#                                   )
cur_path = os.path.dirname(os.path.realpath(__file__))
file_path = f"{cur_path}/../DataAPI/MockAccountConfig"
client_config = TigerOpenClientConfig(props_path=file_path)
quote_client = QuoteClient(client_config)

SYMBOL = "symbol"
TIME = "time"
CLOSE = "close"
DATE = "time_str"

# 需要请求的k线bar的总个数.  total number of requested bars.
BARS_TOTAL_NUMBER = 700000

# 每次请求bar的个数，系统限制每个symbol每次最多请求1200个bar.
# number of bars per request, the system limits each symbol to a maximum of 1200 bars per request.
BARS_BATCH_SIZE = 300

# 每次请求symbol的个数，系统限制每次最多请求50个symbol
# number of symbols per request, the system limits each request to a maximum of 50 symbols.
SYMBOLS_BATCH_SIZE = 50

# 每次请求的间隔时间，防止过快的请求频率触发系统限流. 单位：秒
# The interval between each request, to prevent requests too fast to trigger the system rate limit. Time unit: second
REQUEST_INTERVAL = 3

# 请求数据的时间频率
time_period = 5


class QuoteExamples(unittest.TestCase):
    def setUp(self) -> None:
        print(f'quote permissions:{quote_client.grab_quote_permission()}')

    def test_get_history(self):
        """
        批量请求历史数据示例. Example of a batch request for historical data
        处理后的历史数据打印后如下.  The processed historical data prints out like this:

        all history:
                                                           time    open    high     low   close  volume
        date                      symbol
        2022-02-11 15:41:00+08:00 00700   1644565260000  476.60  477.20  476.60  477.20   49200
                                  01810   1644565260000   16.56   16.60   16.56   16.56  408000
        2022-02-11 15:42:00+08:00 00700   1644565320000  477.00  477.20  476.20  476.20   95500
                                  01810   1644565320000   16.56   16.58   16.54   16.56  199200
        2022-02-11 15:43:00+08:00 00700   1644565380000  476.20  476.40  475.40  475.40   73110
        ...                                         ...     ...     ...     ...     ...     ...
        2022-02-22 11:08:00+08:00 01810   1645499280000   15.62   15.62   15.60   15.62  128400
        2022-02-22 11:09:00+08:00 00700   1645499340000  434.80  435.00  434.20  434.60   83988
                                  01810   1645499340000   15.60   15.64   15.60   15.62  243400
        2022-02-22 11:10:00+08:00 00700   1645499400000  434.60  434.80  434.20  434.60   29399
                                  01810   1645499400000   15.60   15.62   15.58   15.62  234200

        close data:
        date                       symbol
        2022-02-14 09:40:00+08:00  00700     469.00
                                   01810      16.16
        2022-02-14 09:41:00+08:00  00700     468.60
                                   01810      16.06
        2022-02-14 09:42:00+08:00  00700     468.00
                                              ...
        2022-02-22 11:37:00+08:00  01810      15.64
        2022-02-22 11:38:00+08:00  00700     436.00
                                   01810      15.62
        2022-02-22 11:39:00+08:00  00700     436.00
                                   01810      15.64
        Name: close, Length: 4200, dtype: float64

        history of 00700:
                                     date           time   open   high    low  close  volume
        symbol
        00700  2022-02-14 09:40:00+08:00  1644802800000  470.0  470.2  468.6  469.0   61900
        00700  2022-02-14 09:41:00+08:00  1644802860000  469.0  469.4  468.4  468.6   87800
        00700  2022-02-14 09:42:00+08:00  1644802920000  468.6  468.6  467.6  468.0  131770
        00700  2022-02-14 09:43:00+08:00  1644802980000  468.0  468.8  467.8  468.4  115694
        00700  2022-02-14 09:44:00+08:00  1644803040000  468.8  468.8  468.4  468.6   29607
        ...                          ...            ...    ...    ...    ...    ...     ...
        00700  2022-02-22 11:35:00+08:00  1645500900000  436.4  436.4  436.0  436.0   39400
        00700  2022-02-22 11:36:00+08:00  1645500960000  436.2  436.2  435.8  436.0   47000
        00700  2022-02-22 11:37:00+08:00  1645501020000  436.0  436.4  435.8  436.2   40802
        00700  2022-02-22 11:38:00+08:00  1645501080000  436.2  436.4  436.0  436.0   43102
        00700  2022-02-22 11:39:00+08:00  1645501140000  436.0  436.2  436.0  436.0    1700

        [2100 rows x 7 columns]

        close of 00700:
        symbol
        00700    469.0
        00700    468.6
        00700    468.0
        00700    468.4
        00700    468.6
                 ...
        00700    436.0
        00700    436.0
        00700    436.2
        00700    436.0
        00700    436.0
        Name: close, Length: 2100, dtype: float64
        """
        # HK market
        symbol1 = 'MNQmain'
        symbols = [symbol1]
        timezone = 'US/Eastern'

        # US market
        # symbol1 = 'AAPL'
        # symbols = [symbol1, 'TSLA']

        end = int(datetime.datetime.today().timestamp() * 1000)
        history = pd.DataFrame()
        for i in range(0, BARS_TOTAL_NUMBER, BARS_BATCH_SIZE):
            if i + BARS_BATCH_SIZE <= BARS_TOTAL_NUMBER:
                limit = BARS_BATCH_SIZE
            else:
                limit = i + BARS_BATCH_SIZE - BARS_TOTAL_NUMBER
            end_time = datetime.datetime.fromtimestamp(end/1000, pytz.timezone(timezone))
            print(f'query {len(symbols)} symobls history, end_time:{end} -- {end_time}, limit:{limit}')
            # 此处请求分钟k线，其他周期可修改period参数.
            # This request is for the minute k line, for other periods, can change 'period' parameter
            part = self._request_bars(symbols=symbols, period=BarPeriod.FIVE_MINUTES, end_time=end, bars_batch_size=BARS_BATCH_SIZE)
            part[TIME] = part[TIME] + time_period * 60 * 1000
            part[DATE] = pd.to_datetime(part[TIME], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone).dt.strftime('%Y-%m-%d %H:%M:%S')
            end = min(part[TIME]) - time_period * 60 * 1000  # 如果不减一下，有一条会重复
            history = pd.concat([history, part], axis=0)
            # old_df = pd.read_csv('data/MNQmain_5min_kline_for_chan_timeoffset.csv')
            # new_df = history[history.time_str>max(old_df.time_str)][['time_str', 'open', 'high', 'low', 'close', 'volume']].iloc[::-1]
            # concat_df = pd.concat([old_df, new_df], axis=0).reset_index(drop=True)
            # concat_df.to_csv('data/MNQmain_5min_kline_for_chan_timeoffset.csv', index=False)
        history.set_index([DATE, SYMBOL], inplace=True)
        history.sort_index(inplace=True)
        print(f'all history:\n{history}')

        # 取收盘价
        close_data = history[CLOSE]
        print(f'close data:\n{close_data}')

        # 取某个symbol的数据
        hist_of_symbol1 = history.reset_index().set_index(SYMBOL).loc[symbol1]
        close_of_symbol1 = hist_of_symbol1[CLOSE]
        print(f'history of {symbol1}:\n {hist_of_symbol1}')
        print(f'close of {symbol1}:\n{close_of_symbol1}')
        return history

    @staticmethod
    def _request_bars(symbols, period, end_time, bars_batch_size):
        """
        请求k线. Request history bars.
        :param symbols: like ['AAPL', 'TSLA']
        :param period: k线周期. tigeropen.common.consts.BarPeriod. like BarPeriod.DAY
        :param end_time: end time in timestamp format. like 1645499400000
        :param bars_batch_size: 每个symbol限制请求的bar数量. bars limit size of each symbol
        :return:
        """
        symbols = list(symbols)
        result = pd.DataFrame()
        for i in range(0, len(symbols), SYMBOLS_BATCH_SIZE):
            part = symbols[i:i + SYMBOLS_BATCH_SIZE]
            quote = quote_client.get_future_bars(part, period=period, end_time=end_time, limit=bars_batch_size)
            result = pd.concat([result, quote], axis=0)
            # to avoid rate limit
            time.sleep(REQUEST_INTERVAL)
        return result
