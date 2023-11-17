#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 14:58
# @Author  : rockieluo
# @File    : crontab_image.py
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


for k, v in code_dict.items():
    try:
        cal_chan_image(k, save_image_path='../TestImage/feishu')
    except Exception as e:
        print('Stock Info Update Wrong: ', k, v, e)

