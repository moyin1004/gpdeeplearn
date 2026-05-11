# coding=utf8
# 用matplotlib绘制一个柱状图分析3部电影3天的票房。
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# 准备
real_names = ["人在囧途", "阿甘正传", "熊出没"]
real_num1 = [5453, 7548, 6543]  # 人在囧途3天票房数据
real_num2 = [1840, 4013, 3421]  # 阿甘正传3天票房数据
real_num3 = [1080, 1673, 2342]  # 熊出没3天票房数据