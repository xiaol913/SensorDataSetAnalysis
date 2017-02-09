# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 绘图使用 ggplot 的 style
plt.style.use('ggplot')


# 数据标准化
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


# 绘图
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


# 为给定的行为画出一段时间（180 × 50ms）的波形图
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=9, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['accelerometerX'], 'accelerometerX')
    plot_axis(ax1, data['timestamp'], data['accelerometerY'], 'accelerometerY')
    plot_axis(ax2, data['timestamp'], data['accelerometerZ'], 'accelerometerZ')
    plot_axis(ax3, data['timestamp'], data['GyroscopeX'], 'GyroscopeX')
    plot_axis(ax4, data['timestamp'], data['GyroscopeY'], 'GyroscopeY')
    plot_axis(ax5, data['timestamp'], data['GyroscopeZ'], 'GyroscopeZ')
    plot_axis(ax6, data['timestamp'], data['GravityX'], 'GravityX')
    plot_axis(ax7, data['timestamp'], data['GravityY'], 'GravityY')
    plot_axis(ax8, data['timestamp'], data['GravityZ'], 'GravityZ')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


data_set = pd.read_csv('all_data.csv')

data_set['accelerometerX'] = feature_normalize(data_set['accelerometerX'])
data_set['accelerometerY'] = feature_normalize(data_set['accelerometerY'])
data_set['accelerometerZ'] = feature_normalize(data_set['accelerometerZ'])
data_set['GyroscopeX'] = feature_normalize(data_set['GyroscopeX'])
data_set['GyroscopeY'] = feature_normalize(data_set['GyroscopeY'])
data_set['GyroscopeZ'] = feature_normalize(data_set['GyroscopeZ'])
data_set['GravityX'] = feature_normalize(data_set['GravityX'])
data_set['GravityY'] = feature_normalize(data_set['GravityY'])
data_set['GravityZ'] = feature_normalize(data_set['GravityZ'])

# 为每个行为绘图
for activity in np.unique(data_set["Activity"]):
    subset = data_set[data_set["Activity"] == activity][:800]
    plot_activity(activity, subset)
