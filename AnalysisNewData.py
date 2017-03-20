# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 绘图使用 ggplot 的 style
plt.style.use('ggplot')


# 处理数据
def analysis_data(data_set):
    data_set['Accelerometer'] = data_set['AccelerometerX'] ** 2 + \
                                data_set['AccelerometerY'] ** 2 + \
                                data_set['AccelerometerZ'] ** 2
    data_set['Gyroscope'] = data_set['GyroscopeX'] ** 2 + data_set['GyroscopeY'] ** 2 + data_set['GyroscopeZ'] ** 2
    data_set['Gravity'] = data_set['GravityX'] ** 2 + data_set['GravityY'] ** 2 + data_set['GravityZ'] ** 2
    return data_set


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['Timestamp'], data['Accelerometer'], 'Accelerometer')
    plot_axis(ax1, data['Timestamp'], data['Gyroscope'], 'Gyroscope')
    plot_axis(ax2, data['Timestamp'], data['Gravity'], 'Gravity')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


# 绘图
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


data_set = pd.read_csv('raw_data.csv')
data_set = analysis_data(data_set)

# 为每个行为绘图
for activity in np.unique(data_set["Activity"]):
    subset = data_set[data_set["Activity"] == activity][:500]
    if activity == 0:
        activity = "OnFeet"
    elif activity == 1:
        activity = "Still"
    elif activity == 2:
        activity = "InVehicle"

    plot_activity(activity, subset)
