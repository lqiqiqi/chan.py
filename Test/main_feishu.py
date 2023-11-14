from datetime import datetime, timedelta, timezone

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

import os
import shutil
import json
import requests
from requests_toolbelt import MultipartEncoder


code_dict = {
    'HK.00700': '腾讯',
    'HK.800000': '恒生指数',
    'SH.513300': '纳指etf',
    'HK.09868': '小鹏',
    'HK.03690': '美团',
    'HK.09618': '京东',
    'SH.000991': '全指医药',
    'SH.000922': '中证红利指数',
    'SH.000905': '中证500指数',
    'SH.512880': '证券',
    # 'HK.01681': '康臣药业'
}

# code = 'HK.00700'
# code = 'HK.07333' # 做空沪深300


def post_url_with_header(url, data):
    try:
        headers = {'Content-Type': 'application/json; chartset=utf-8'}
        response = requests.post(url=url, headers=headers, data=data, timeout=300)
        if response.status_code != 200:
            # logger.error(response)
            return {'result': False, 'message': u'请求失败' % response.status_code}
        return json.loads(response.text)
    except Exception as e:
        return {'result': False, 'message': u'请求异常' % e}


def send_msg(content, type):
    url = "https://open.feishu.cn/open-apis/bot/v2/hook/aa8d6a93-9517-4fb4-b93f-8c98d8a03a77"
    if type == 'text':
        data = {"msg_type":"text", "content": {"text": content}}
    elif type == 'image':
        data = {
            "msg_type": "image",
            "content": {
                "image_key": content
            }
        }
    res = post_url_with_header(url, json.dumps(data))
    return res


def get_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = {
        "app_id": "cli_a5cf12f52dba5013",
        "app_secret": "hpmzjMychWHi9D6ApbaS3e7NwTD1Hnyn"
    }
    res = post_url_with_header(url=url, data=json.dumps(data))
    if res['code'] == 0:
        return res['tenant_access_token']
    return None


def upload_image(image_path, access_token):
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    form = {'image_type': 'message',
            'image': (open(image_path, 'rb'))}  # 需要替换具体的path
    multi_form = MultipartEncoder(form)
    headers = {
        'Authorization': f'Bearer {access_token}',  ## 获取tenant_access_token, 需要替换为实际的token
    }
    headers['Content-Type'] = multi_form.content_type
    response = requests.request("POST", url, headers=headers, data=multi_form)
    # print(response.headers['X-Tt-Logid'])  # for debug or oncall
    return response.content


def cal_chan_image(code):
    # 创建北京时区对象
    beijing_tz = timezone(timedelta(hours=8))
    # 获取北京时间的当前时间
    now_beijing = datetime.now(beijing_tz)
    begin_date = (now_beijing - timedelta(days=300)).strftime('%Y-%m-%d')
    now_date = now_beijing.strftime('%Y-%m-%d')

    begin_time = begin_date
    end_time = now_date
    data_src = DATA_SRC.FUTU
    lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_30M, KL_TYPE.K_5M]

    config = CChanConfig({
        "bi_strict": False,
        "bi_fx_check": "loss",
        "bi_algo": "normal",
        "bi_end_is_peak": False,
        "one_bi_zs": False,
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
        # "zs_algo": "over_seg"
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_segseg": False,
        "plot_segzs": True,
        "plot_zs": False,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
        "plot_segbsp": True,
        "plot_boll": True
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
            "sub_lv_cnt": 30,
            "facecolor": 'green'
        },
        "figure": {
            "x_range": 10000,
            "only_top_lv": False
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

    if not config.triger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        # plot_driver.figure.show()
        image_path = f'../TestImage/feishu/{code.split(".")[-1]}.jpg'
        plot_driver.figure.savefig(image_path)

    return image_path


if __name__ == "__main__":
    access_token = get_token()
    for k, v in code_dict.items():
        image_path = cal_chan_image(k)
        res = upload_image(image_path, access_token)
        res = json.loads(res)
        if res['code'] == 0:
            send_msg(v, type='text')
            send_msg(res['data']['image_key'], type='image')

    # # 删除文件夹中的图片
    # folder_path = '../TestImage/feishu/'  # 将这里替换为你的文件夹路径
    #
    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    #     elif os.path.isdir(file_path):
    #         shutil.rmtree(file_path)
