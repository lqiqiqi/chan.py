#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 16:28
# @Author  : rockieluo
# @File    : VOC.py


class VOC:
    def __init__(self):
        super(VOC, self).__init__()
        self.vol_arr = []
        self.period = 5

    def add(self, vol):
        """
        计算成交量变化
        """
        if not len(self.vol_arr):
            self.vol_arr.append(vol)
        self.vol_arr.append(vol)
        if len(self.vol_arr) > self.period:
            self.vol_arr.pop(0)
        voc = (self.vol_arr[-1] - self.vol_arr[-2]) / self.vol_arr[-2] * 100
        return voc
