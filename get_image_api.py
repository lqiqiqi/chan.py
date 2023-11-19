from datetime import datetime, timedelta, timezone

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from DataAPI.FuTuAPI import Futu
from Debug.check_is_1_3_bp import check_is_1_3_bsp
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

import os
import shutil
import json
import requests
from requests_toolbelt import MultipartEncoder

from flask import Flask, jsonify, request
from flask_httpauth import HTTPBasicAuth

# from flask_cors import CORS


from Test.config import chan_config, plot_config, plot_para

app = Flask(__name__)
# CORS(app)
auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    # 在此处添加您的验证逻辑，例如检查数据库中的用户名和密码
    if username == 'user' and password == '123456':
        return username
    return None


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
        data = {"msg_type": "text", "content": {"text": content}}
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


def cal_chan_image(code, save_image_path='./TestImage/feishu'):
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

    if not config.triger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        # plot_driver.figure.show()
        image_path = f'{save_image_path}/{code.split(".")[-1]}.jpg'
        plot_driver.figure.savefig(image_path)

    return image_path


@app.route('/get_image', methods=['GET'])
@auth.login_required
def get_image_api():
    code = request.args.get('code', '')
    access_token = get_token()
    if code != '':
        try:
            name = Futu.get_stock_name(code)
            image_path = cal_chan_image(code)
            res = upload_image(image_path, access_token)
            res = json.loads(res)
        except Exception as e:
            msg = e
            return jsonify({'message': f'get image failed {msg}'})

        if res['code'] == 0:
            send_msg(name, type='text')
            send_msg(res['data']['image_key'], type='image')

        return jsonify({'message': 'get image success'})
    else:
        return jsonify({'message': 'wrong code'})


@app.route('/check_is_1_3_bsp', methods=['GET'])
@auth.login_required
def check_is_1_3_bsp_api():
    code = request.args.get('code', '')
    date_time = request.args.get('date_time', '')
    if code != '':
        try:
            check_res = check_is_1_3_bsp(code, date_time)
        except Exception as e:
            msg = e
            return jsonify({'message': f'check failed {msg}'})
        return jsonify({'message': check_res})
    else:
        return jsonify({'message': 'wrong code'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
