# -*- coding: utf-8 -*-
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator
import matplotlib.font_manager as font_manager
import json
import os
import numpy as np
import sys

sns.set()


def myplot_csv(file_dir, t_max):
    x = []
    y = []
    for root, dirs, files in os.walk(file_dir, topdown=False):
        for filename in files:
            f = open(root + '/' + filename)
            df = pd.read_csv(f)
            print(df.tail(1))
            x.append(df['Step'])
            y.append(df['Value'])
            h = df['Value'].tolist()
    ave_y = []
    for i in range(len(x)):
        temp = np.interp(range(1000, t_max, step_), x[i], y[i])  # 线性插值
        ave_y.append(temp)

    ave_y = np.array(ave_y)

    ave = np.mean(ave_y, axis=0)
    ave_max = np.max(ave_y, axis=0)
    ave_min = np.min(ave_y, axis=0)
    # return ave, ave_max, ave_min
    res_max = (ave_max - ave_min) / 4 * 3 + ave_min
    res_min = (ave_max - ave_min) / 4 + ave_min
    return ave, res_max, res_min


def myplot_tblog(file_dir, key, t_max):
    x = []
    y = []
    for root, dirs, files in os.walk(file_dir, topdown=False):
        if os.path.exists(root + '/step.npy'):
            x.append(np.load(root + '/step.npy'))
            y.append(np.load(root + '/win.npy'))
    ave_y = []
    for i in range(len(x)):
        temp = np.interp(range(1000, t_max, step_), x[i], y[i])  # 线性插值
        ave_y.append(temp)

    ave_y = np.array(ave_y)

    ave = np.mean(ave_y, axis=0)
    ave_max = np.max(ave_y, axis=0)
    ave_min = np.min(ave_y, axis=0)
    # return ave, ave_max, ave_min
    res_max = (ave_max - ave_min) / 5 * 4 + ave_min
    res_min = (ave_max - ave_min) / 4 + ave_min
    return ave, res_max, res_min


def myplot_cout(file_dir, key, smooth_rate=0.75):
    step_list = []
    value_list = []
    len_ = 1000000
    t_max = 1e9
    # print(file_dir)
    last_v, last_step = None, None
    for root, dirs, files in os.walk(file_dir, topdown=False):
        # print(root,dirs,files)
        if len(dirs) == 0:
            with open(f'{root}/cout.txt', 'r') as f:
                lines = f.readlines()
            step = []
            v = []
            s = None
            for l in lines:
                Obj = l.split()
                if 'Recent' in Obj:
                    try:
                        s = eval(Obj[7])
                    except:
                        print(Obj)
                        print(Obj[7])
                else:
                    for i in range(len(Obj)):
                        if Obj[i] == f'{key}:' and Obj[i + 1] != 'nan':
                            v.append(eval(Obj[i + 1]))
                            if s is not None:
                                step.append(s)
                            break
            last_v, last_step = v, step
            assert len(step) == len(v)
            if len(step) == 0:
                continue
            else:
                print(step[-1], v[-1])
                # if step[-1]<=0.8*t_max or v[-1]==0.0:continue
                # if step[-1]<=100000:continue
                t_max = min(t_max, step[-1])
            value_list.append(v)
            step_list.append(step)

    if not value_list:
        value_list.append(last_v)
        step_list.append(last_step)
    ave_y = []

    for i in range(len(value_list)):
        temp = np.interp(range(1000, t_max, step_), step_list[i], value_list[i])  # 线性插值
        ave_y.append(temp)

    # 是否对数据进行平滑处理
    if smooth_rate:
        new_ave_y = []
        for index in range(len(ave_y)):
            ave_y_index = ave_y[index]
            smoothed_ave_y = []
            weight = smooth_rate
            last = 0
            for point in ave_y_index:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed_ave_y.append(smoothed_val)
                last = smoothed_val
            new_ave_y.append(np.array(smoothed_ave_y))
        ave_y = new_ave_y
    else:
        ave_y = np.array(ave_y)

    ave = np.mean(ave_y, axis=0)
    ave_max = np.max(ave_y, axis=0)
    ave_min = np.min(ave_y, axis=0)
    # return ave, ave_max, ave_min,t_max
    res_max = (ave_max - ave_min) / 4 * 3 + ave_min
    res_min = (ave_max - ave_min) / 4 + ave_min

    return ave, res_max, res_min, t_max


def formatnum_x(x, pos):
    return '%.1f' % (x / 1000000.0)


def formatnum_y(x, pos):
    return '%i' % (x * 100)


formatter_x = FuncFormatter(formatnum_x)
font = font_manager.FontProperties(family='sans-serif', weight='normal', size=15)
alg_l_cout = ['vdn_env=8_adam_td_lambda']
color = ['darkorange', 'green', 'blue', 'darkorchid', 'steelblue', 'red', 'c', 'y', 'm', 'deepskyblue', 'gray', 'pink',
         'k']
map_l = ['3m']
color_hash = {
    'VDN': 'darkorange',
    'QMIX': 'red',
    'QPLEX': 'deepskyblue',
    'divide_no_PER': 'red',
    'greedy_0.8_32': 'c',
    'greedy_0.9_32': 'y',
    'greedy_1_32': 'm',
    'greedy_0.8_40': 'green',
    'PER_1_32_0.0005': 'c',
    'PER_0.9_32_0.0005': 'y',
    'PER_0.8_32_0.0005': 'm',
    'PER_0.8_40_0.0005': 'green',
    'PER_0.8_32_0.0001': 'darkorange',
    'PER_0.8_40_0.0001': 'm',
    'PER_0.9_32_0.0001': 'deepskyblue',
    'PER_hard_0.8_32_0.0005': 'blue',
    'PER_hard_0.8_40_0.0005': 'deepskyblue',
    'PER_hard_0.8_32_0.0001': 'green',
    'PER_hard_0.8_40_0.0001': 'darkorange',

}

# smooth_rate = 0时不进行平滑处理

t_max = 7000000
step_ = 15000
linewidth = 3
smooth_rate = 0.75

t_max = 2000000
step_ = 1
linewidth = 2
smooth_rate = 0.2

color_index = -1
fontsize = 10
f, ax = plt.subplots(figsize=(12, 6))

ax.grid(True, alpha=0.7)
plt.xlim(0, t_max)

color_index = -1

# key = 'return_mean'
key = 'return_mean'
# key = 'battle_win_mean'

# filename = '/home/ubuntu/Hok_Marl_ppo/results/sacred/battle5v5/mappo_env_adjustpolicy_loss_withmodel2/1'
filename = '/home/ubuntu/Marl_Huaru/results/sacred/battle5v5/mappo_modified_to_ippo/29'
if os.path.exists(filename):
    color_index = 2
    ave, ave_max, ave_min, t_max = myplot_cout(filename, key, smooth_rate)
    ax.plot(range(1000, t_max, step_), ave, lw=linewidth, color=color[color_index], label='mappo')
    ax.fill_between(range(1000, t_max, step_), ave_max, ave_min,
                    color=color[color_index],  # 颜色
                    alpha=0.1)  # 透明度

# filename = '/home/ubuntu/Hok_Marl_ppo/results/sacred/battle5v5/before_result/1'
# if os.path.exists(filename):
#     color_index = 2
#     ave2, ave_max2, ave_min2, t_max2 = myplot_cout(filename, key, smooth_rate)
#     # ax.plot(range(1000, t_max, step_), ave, lw=linewidth, color=color[color_index], label='mappo')
#     # ax.fill_between(range(1000, t_max, step_), ave_max, ave_min,
#     #                 color=color[color_index],  # 颜色
#     #                 alpha=0.1)  # 透明度
#
# ave = np.append(ave, ave2)
# ave_max = np.append(ave_max, ave_max2)
# ave_min = np.append(ave_min, ave_min2)
# t_max += t_max2
# print(ave.shape)
# print(t_max)
# ax.plot(range(0, t_max-2000, step_), ave, lw=linewidth, color=color[color_index], label='mappo')
# ax.fill_between(range(0, t_max-2000, step_), ave_max, ave_min,
#                 color=color[color_index],  # 颜色
#                 alpha=0.1)  # 透明度


plt.xlabel('Timesteps(M)', fontsize=12)
# plt.ylabel('Reward Mean', fontsize=12)
plt.ylabel('Return Mean', fontsize=12)
# plt.title('3s_vs_5z(Hard)', fontsize=18)
# plt.title('Reward in HuaruBattle', fontsize=12)
plt.title('Result in HuaruBattle', fontsize=12)
# plt.title('3s5zvs3s6z(Super Hard)_to_3svs5z(Hard)', fontsize=18)
# upper left or lower right
leg = plt.legend(fontsize=1, loc='best', framealpha=0, prop=font)
leg_lines = leg.get_lines()
plt.legend()
# plt.show()
plt.savefig(f'/home/ubuntu/Marl_Huaru/results/sacred/battle5v5/mappo_modified_to_ippo/29/result_{key}.jpg', bbox_inches='tight')
# plt.close()
