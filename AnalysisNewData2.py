# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 绘图使用 ggplot 的 style
plt.style.use('ggplot')


# normalize
def normalize_data(data_set):
    mu = np.mean(data_set, axis=0)
    sigma = np.std(data_set, axis=0)
    return (data_set - mu) / sigma


# 处理数据
def analysis_data(data_set):
    data_set['LinearX'] = data_set['AccelerometerX'] - data_set['GravityX']
    data_set['LinearY'] = data_set['AccelerometerY'] - data_set['GravityY']
    data_set['LinearZ'] = data_set['AccelerometerZ'] - data_set['GravityZ']
    data_set['AccelerometerX'] = normalize_data(data_set['AccelerometerX'])
    data_set['AccelerometerY'] = normalize_data(data_set['AccelerometerY'])
    data_set['AccelerometerZ'] = normalize_data(data_set['AccelerometerZ'])
    data_set['LinearX'] = normalize_data(data_set['LinearX'])
    data_set['LinearY'] = normalize_data(data_set['LinearY'])
    data_set['LinearZ'] = normalize_data(data_set['LinearZ'])
    data_set['GravityX'] = normalize_data(data_set['GravityX'])
    data_set['GravityY'] = normalize_data(data_set['GravityY'])
    data_set['GravityZ'] = normalize_data(data_set['GravityZ'])
    return data_set


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=9, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['Timestamp'], data['AccelerometerX'], 'AccelerometerX')
    plot_axis(ax1, data['Timestamp'], data['AccelerometerY'], 'AccelerometerY')
    plot_axis(ax2, data['Timestamp'], data['AccelerometerZ'], 'AccelerometerZ')
    plot_axis(ax3, data['Timestamp'], data['LinearX'], 'LinearX')
    plot_axis(ax4, data['Timestamp'], data['LinearY'], 'LinearY')
    plot_axis(ax5, data['Timestamp'], data['LinearZ'], 'LinearZ')
    plot_axis(ax6, data['Timestamp'], data['GravityX'], 'GravityX')
    plot_axis(ax7, data['Timestamp'], data['GravityY'], 'GravityY')
    plot_axis(ax8, data['Timestamp'], data['GravityZ'], 'GravityZ')
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


def split_data(data_set):
    result = {}
    for i in range(0, 250):
        result[i] = data_set[i * 500: (i + 1) * 500]
    return result


data_set = pd.read_csv('./data_set/raw_data.csv')
# data_set = analysis_data(data_set)

# 为每个行为绘图
# for activity in np.unique(data_set["Activity"]):
#     subset = data_set[data_set["Activity"] == activity][:500]
#     if activity == 0:
#         activity = "OnFeet"
#     elif activity == 1:
#         activity = "Still"
#     elif activity == 2:
#         activity = "InVehicle"
#
#     plot_activity(activity, subset)

del data_set['Unnamed: 0']
# del data_set['Timestamp']
feet_data = split_data(analysis_data(data_set[data_set["Activity"] == 0][:125000]))
still_data = split_data(analysis_data(data_set[data_set["Activity"] == 1][:125000]))
vehicle_data = split_data(analysis_data(data_set[data_set["Activity"] == 2][:125000]))

df = pd.DataFrame()
df = df.append(feet_data, ignore_index=True)
df = df.append(still_data, ignore_index=True)
df = df.append(vehicle_data, ignore_index=True)

# plot_activity("Feet", df[1][0])
print(df[2][0])