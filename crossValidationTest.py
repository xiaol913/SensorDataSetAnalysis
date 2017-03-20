import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def double_data(data_set):
    return data_set ** 2


# analysis data
def new_function(data_set):
    data_set['AccelerometerX'] = abs(data_set['AccelerometerX'])
    data_set['AccelerometerY'] = abs(data_set['AccelerometerY'])
    data_set['AccelerometerZ'] = abs(data_set['AccelerometerZ'])
    data_set['Accelerometer_value'] = data_set['AccelerometerX'] ** 2 + \
                                      data_set['AccelerometerY'] ** 2 + \
                                      data_set['AccelerometerZ'] ** 2
    data_set['GyroscopeX'] = abs(data_set['GyroscopeX'])
    data_set['GyroscopeY'] = abs(data_set['GyroscopeY'])
    data_set['GyroscopeZ'] = abs(data_set['GyroscopeZ'])
    data_set['Gyroscope_value'] = data_set['GyroscopeX'] ** 2 + \
                                      data_set['GyroscopeY'] ** 2 + \
                                      data_set['GyroscopeZ'] ** 2
    data_set['GravityX'] = abs(data_set['GravityX'])
    data_set['GravityY'] = abs(data_set['GravityY'])
    data_set['GravityZ'] = abs(data_set['GravityZ'])
    data_set['Gravity_value'] = data_set['GravityX'] ** 2 + \
                                      data_set['GravityY'] ** 2 + \
                                      data_set['GravityZ'] ** 2
    return data_set


data = pd.read_csv("raw_data.csv")
data = new_function(data)
X = data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
          'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'Gyroscope_value',
          'GravityX', 'GravityY', 'GravityZ', 'Gravity_value']]
y = data[['Activity']]

clf = LogisticRegression()
predictions = cross_val_predict(clf, X, y.values.ravel(), cv=5)
print(metrics.accuracy_score(y.values.ravel(), predictions))
