# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# normalize
def feature_normalize(data_set):
    mu = np.mean(data_set, axis=0)
    sigma = np.std(data_set, axis=0)
    return (data_set - mu) / sigma


# read data and normalize it.
def read_data(data_set):
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


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
result_data = pd.read_csv('result_data.csv')

train_data = read_data(train_data)
test_data = read_data(test_data)

# training data
train = train_data.filter(regex='Accelerometer*|Gyroscope*|Gravity*|Activity')
train_np = train.values
y = train_np[:, -1]
X = train_np[:, :-1]

# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf = RandomForestClassifier()
clf.fit(X, y)

test = test_data.filter(regex='Accelerometer*|Gyroscope*|Gravity*')
predictions = clf.predict(test)
result = pd.DataFrame({'Activity': predictions.astype(np.int32)})
result.to_csv('result.csv', index=False)

# compute the matching rate
count = 0
for i in range(0, len(result)):
    if result['Activity'][i] == result_data['Activity'][i]:
        count += 1

print float(count) / float(len(result))
