# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

# 绘图使用 ggplot 的 style
# plt.style.use('ggplot')


# 数据标准化
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


# # 绘图
# def plot_axis(ax, x, y, title):
#     ax.plot(x, y)
#     ax.set_title(title)
#     ax.xaxis.set_visible(False)
#     ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
#     ax.set_xlim([min(x), max(x)])
#     ax.grid(True)
#
#
# # 为给定的行为画出一段时间（180 × 50ms）的波形图
# def plot_activity(activity, data):
#     fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=9, figsize=(15, 10), sharex=True)
#     plot_axis(ax0, data['timestamp'], data['accelerometerX'], 'accelerometerX')
#     plot_axis(ax1, data['timestamp'], data['accelerometerY'], 'accelerometerY')
#     plot_axis(ax2, data['timestamp'], data['accelerometerZ'], 'accelerometerZ')
#     plot_axis(ax3, data['timestamp'], data['GyroscopeX'], 'GyroscopeX')
#     plot_axis(ax4, data['timestamp'], data['GyroscopeY'], 'GyroscopeY')
#     plot_axis(ax5, data['timestamp'], data['GyroscopeZ'], 'GyroscopeZ')
#     plot_axis(ax6, data['timestamp'], data['GravityX'], 'GravityX')
#     plot_axis(ax7, data['timestamp'], data['GravityY'], 'GravityY')
#     plot_axis(ax8, data['timestamp'], data['GravityZ'], 'GravityZ')
#     plt.subplots_adjust(hspace=0.2)
#     fig.suptitle(activity)
#     plt.subplots_adjust(top=0.90)
#     plt.show()


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

data_test = pd.read_csv('all_data_test.csv')

data_test['accelerometerX'] = feature_normalize(data_test['accelerometerX'])
data_test['accelerometerY'] = feature_normalize(data_test['accelerometerY'])
data_test['accelerometerZ'] = feature_normalize(data_test['accelerometerZ'])
data_test['GyroscopeX'] = feature_normalize(data_test['GyroscopeX'])
data_test['GyroscopeY'] = feature_normalize(data_test['GyroscopeY'])
data_test['GyroscopeZ'] = feature_normalize(data_test['GyroscopeZ'])
data_test['GravityX'] = feature_normalize(data_test['GravityX'])
data_test['GravityY'] = feature_normalize(data_test['GravityY'])
data_test['GravityZ'] = feature_normalize(data_test['GravityZ'])

# 为每个行为绘图
# for activity in np.unique(data_set["Activity"]):
#     subset = data_set[data_set["Activity"] == activity][:800]
#     plot_activity(activity, subset)

# print data_test

train = data_set.filter(regex='Activity|accelerometer*|Gyroscope*|Gravity*')
train_np = train.values
y = train_np[:, -1]
x = train_np[:, :-1]

# print x

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x, y)

test = data_test.filter(regex='accelerometer*|Gyroscope*|Gravity*')
predictions = clf.predict(test)
result = pd.DataFrame({'Activity': predictions.astype(np.int32)})
result.to_csv('result.csv', index=False)
