#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 14:45
# @Author  : rockieluo
# @File    : app.py

import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from Test.main_feishu import cal_chan_image

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
    'SH.512880': '证券'
}


req_dash_prefix = os.getenv('REQ_DASH_PREFIX')
rts_dash_prefix = os.getenv('RTS_DASH_PREFIX')

# app server
app = dash.Dash('DemoDashBoard', requests_pathname_prefix=req_dash_prefix, routes_pathname_prefix=rts_dash_prefix,
                assets_folder='TestImage', assets_url_path='TestImage')
# app = dash.Dash(__name__)

server = app.server

# 图片的本地路径或URL
# image_options = [
#     {"label": "图片 1", "value": "./TestImage/feishu/00700.jpg"},
#     {"label": "图片 2", "value": "./TestImage/feishu/800000.jpg"}
#     # {"label": "图片 3", "value": "https://example.com/path/to/your/image3.jpg"},
# ]
image_options = [
    {"label": code, "value": f"./TestImage/feishu/{code.split('.')[-1]}.jpg"} for code, _ in code_dict.items()
]

# 创建 Dash 布局
app.layout = html.Div([
    dcc.Dropdown(
        id="image-dropdown",
        options=image_options,
        value=image_options[0]["value"],
    ),
    html.Div(id="image-container")
])

# 定义回调函数
@app.callback(
    Output("image-container", "children"),
    [Input("image-dropdown", "value")]
)
def update_image(selected_image):
    return html.Img(src=selected_image, style={"width": "auto", "height": "auto"})


if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=80)
