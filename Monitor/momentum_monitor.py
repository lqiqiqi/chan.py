import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

from get_image_api import send_msg


def calculate_momentum(series, window=21):
    """滚动窗口动量计算（支持向量化操作）"""
    if len(series) < window:
        return np.nan
    X = np.arange(window).reshape(-1, 1)
    series = np.where(series[-window:] <= 0, 1e-9, series[-window:])
    log_series = np.log(series).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, log_series)
    return (np.exp(model.coef_[0][0] * 252) - 1) * model.score(X, log_series)


# 配置参数
SYMBOLS = ["XLF", "XLP", "XLE", "XLI", "XLV", "XLU",
           "SMH", "XLB", "XLK", "XLY", "KBE", "VNQ",
           "IBB", "IYF", "KIE", "GLD",     "SPY", "KWEB", "XLF", "IBIT", "QQQ", "TLT", "IWM",
            "IEMG", "RSP", "IEF", "IGV", "SCHG", "FXI",
            "IJH", "IEFA", "LQD", '510300.SS', '510880.SS', '159915.SZ', '510500.SS', '159928.SZ']
LOOKBACK_WINDOW = 21  # 动量计算窗口
TREND_DAYS = 5  # 排名趋势观察期

# 数据获取
end_date = datetime.today()
data = yf.download(
    SYMBOLS,
    start=(end_date - timedelta(LOOKBACK_WINDOW + TREND_DAYS + 30)).strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    group_by='ticker',
    auto_adjust=True,
    progress=False
)

# 构建动量矩阵
momentum_history = []
for symbol in SYMBOLS:
    try:
        df = data[symbol][['Close', 'Volume']].dropna()
        dates = df.index

        # 向量化计算动量
        price_momentum = [calculate_momentum(df['Close'].values[:i + 1])
                          for i in range(LOOKBACK_WINDOW, len(df))]
        # volume_momentum = [calculate_momentum(df['Volume'].values[:i + 1])
        #                    for i in range(LOOKBACK_WINDOW, len(df))]

        # 对齐日期索引
        momentum_df = pd.DataFrame({
            'price': price_momentum,
            # 'volume': volume_momentum
        }, index=dates[LOOKBACK_WINDOW:])
        momentum_df['symbol'] = symbol

        momentum_history.append(momentum_df)
    except KeyError:
        continue

momentum_df = pd.concat(momentum_history)

# 计算每日双维度排名
momentum_df['price_rank'] = momentum_df.groupby(level=0)['price'].rank(ascending=False)
# momentum_df['volume_rank'] = momentum_df.groupby(level=0)['volume'].rank(ascending=False)


# 计算排名变化趋势
def calculate_trend(group):
    return group.rolling(TREND_DAYS).apply(
        lambda x: (x[-1] - x[0]) if len(x) == TREND_DAYS else np.nan)


momentum_df['rank_trend'] = momentum_df.groupby('symbol')['price_rank'].transform(calculate_trend)

# 获取最新趋势数据
latest_date = momentum_df.index.max()
result = (
    momentum_df.loc[latest_date]
    .sort_values('rank_trend')
    [['symbol', 'price_rank', 'rank_trend']]
    .dropna()
)

# 生成报告字符串
report = []

# 价格排名前五
top_price = result.nsmallest(10, 'price_rank')  # 假设price_rank值越小排名越高
price_str = "🏆 当前价格动量排名前十：\n" + "\n".join(
    [f"{i+1}. {row.symbol} (排名：{row.price_rank:.2f})"
     for i, (_, row) in enumerate(top_price.iterrows())]
)

# 排名上升前五（rank_trend负值越大表示上升越快）
top_risers = result.nsmallest(5, 'rank_trend')  # 选择rank_trend最小的（最负的）
risers_str = f"\n\n🚀 最近{TREND_DAYS}天排名上升最快前五：\n" + "\n".join(
    [f"{i+1}. {row.symbol} (上升幅度：{abs(row.rank_trend):.2f}位)"
     for i, (_, row) in enumerate(top_risers.iterrows())]
)

# 组合报告
full_report = f"""
📈 {datetime.today().strftime('%Y-%m-%d')} 动量监控报告
{'-'*40}
{price_str}
{risers_str}
"""

print(full_report)
send_msg(full_report, 'text')