#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 16:28
# @Author  : rockieluo
# @File    : VOMA.py

class VOMA_Item:
    def __init__(self, voma10, voma3):
        self.voma10 = voma10
        self.voma3 = voma3
        self.voma_diff = self.voma3 - self.voma10

class VOMA:
    def __init__(self):
        super(VOMA, self).__init__()
        self.vol_arr = []
        self.period = 10

    def add(self, vol):
        """
        计算vol均线
        """
        while len(self.vol_arr) < 10:
            self.vol_arr.append(vol)
        self.vol_arr.append(vol)
        if len(self.vol_arr) > self.period:
            self.vol_arr.pop(0)
        voma10 = sum(self.vol_arr[-10:]) / 10
        voma3 = sum(self.vol_arr[-3:]) / 3
        voma = VOMA_Item(voma10=voma10, voma3=voma3)
        return voma
