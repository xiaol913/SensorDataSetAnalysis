import pandas as pd
import numpy as np
from sklearn.externals import joblib


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


clf = joblib.load('feature.pkl')
result_data = pd.read_csv('result_data.csv')
test_data = pd.read_csv('test_data.csv')
test_data = read_data(test_data)
test = test_data.filter(regex='Accelerometer*|Gyroscope*|Gravity*')
predictions = clf.predict(test)
result = pd.DataFrame({'Activity': predictions.astype(np.int32)})
result.to_csv('result.csv', index=False)

# compute the matching rate
count = 0
for i in range(0, len(result)):
    if result['Activity'][i] == result_data['Activity'][i]:
        count += 1

rate = float(count) / float(len(result))
print(rate)
