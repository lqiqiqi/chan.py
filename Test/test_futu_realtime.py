from futu import *
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

ret_sub, err_message = quote_ctx.subscribe(['HK.00700'], [SubType.K_15M], subscribe_push=False)

# 先订阅 K 线类型。订阅成功后 OpenD 将持续收到服务器的推送，False 代表暂时不需要推送给脚本
if ret_sub == RET_OK:  # 订阅成功
    ret, data = quote_ctx.get_cur_kline('HK.00700', 2, KLType.K_15M, AuType.QFQ)  # 获取港股00700最近2个 K 线数据
    if ret == RET_OK:
        print(data)
        print(data['turnover_rate'][0])   # 取第一条的换手率
        print(data['turnover_rate'].values.tolist())   # 转为 list
    else:
        print('error:', data)
else:
    print('subscription failed', err_message)
quote_ctx.close()  # 关闭当条连接，OpenD 会在1分钟后自动取消相应股票相应类型的订阅
