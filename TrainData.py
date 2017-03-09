# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# normalize
def feature_normalize(data_set):
    mu = np.mean(data_set, axis=0)
    sigma = np.std(data_set, axis=0)
    return (data_set - mu) / sigma


# read data and normalize it.
def normalize_data(data_set):
    data_set['AccelerometerX'] = feature_normalize(data_set['AccelerometerX'])
    data_set['AccelerometerY'] = feature_normalize(data_set['AccelerometerY'])
    data_set['AccelerometerZ'] = feature_normalize(data_set['AccelerometerZ'])
    data_set['GyroscopeX'] = feature_normalize(data_set['GyroscopeX'])
    data_set['GyroscopeY'] = feature_normalize(data_set['GyroscopeY'])
    data_set['GyroscopeZ'] = feature_normalize(data_set['GyroscopeZ'])
    data_set['GravityX'] = feature_normalize(data_set['GravityX'])
    data_set['GravityY'] = feature_normalize(data_set['GravityY'])
    data_set['GravityZ'] = feature_normalize(data_set['GravityZ'])
    return data_set


def double_data(data_set):
    return data_set ** 2


# analysis data
def new_function(data_set):
    data_set['AccelerometerX'] = double_data(data_set['AccelerometerX'])
    data_set['AccelerometerY'] = double_data(data_set['AccelerometerY'])
    data_set['AccelerometerZ'] = double_data(data_set['AccelerometerZ'])
    data_set['GyroscopeX'] = double_data(data_set['GyroscopeX'])
    data_set['GyroscopeY'] = double_data(data_set['GyroscopeY'])
    data_set['GyroscopeZ'] = double_data(data_set['GyroscopeZ'])
    data_set['GravityX'] = double_data(data_set['GravityX'])
    data_set['GravityY'] = double_data(data_set['GravityY'])
    data_set['GravityZ'] = double_data(data_set['GravityZ'])
    return data_set


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
result_data = pd.read_csv('result_data.csv')

# train_data = normalize_data(train_data)
# test_data = normalize_data(test_data)
# train_data = new_function(train_data)
# test_data = new_function(test_data)


# training data
train = train_data.filter(regex='Accelerometer|Activity')
train_np = train.values
y = train_np[:, -1]
X = train_np[:, :-1]
clf = linear_model.RidgeClassifier(alpha=0.001, fit_intercept=True, normalize=True,
                                   copy_X=True, max_iter=1000, tol=0.000001,
                                   class_weight=None, solver='auto', random_state=None)
# 8713
clf.fit(X, y)

test = test_data.filter(regex='Accelerometer')
predictions = clf.predict(test)
result = pd.DataFrame({'Activity': predictions.astype(np.int32)})

# compute the matching rate
count = 0
for i in range(0, len(result)):
    if result['Activity'][i] == result_data['Activity'][i]:
        count += 1

rate = count / len(result)
print(rate)

# save clf
if rate > 0.86:
    result.to_csv('result.csv', index=False)
    joblib.dump(clf, 'feature.pkl')
