from candlestick.patterns.candlestick_finder import CandlestickFinder


class Star(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        return (prev_close > prev_open and # 昨天涨
                abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and # 昨天实体较大
                0.3 > abs(close - open) / (high - low) >= 0.1 and # 今天实体较小
                prev_close < close and # 今天收盘高于昨天收盘
                prev_close < open) # 今天开盘高于昨天收盘
