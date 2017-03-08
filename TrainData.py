# coding=utf-8
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib


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


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
result_data = pd.read_csv('result_data.csv')

train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# training data
train = train_data.filter(regex='Accelerometer*|Gyroscope*|Gravity*|Activity')
train_np = train.values
y = train_np[:, -1]
X = train_np[:, :-1]

clf = linear_model.SGDClassifier(alpha=0.01, average=False, class_weight=None, epsilon=0.1,
                                 eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                                 learning_rate='optimal', loss='hinge', n_iter=10, n_jobs=1,
                                 penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                                 verbose=0, warm_start=False)
clf.fit(X, y)

test = test_data.filter(regex='Accelerometer*|Gyroscope*|Gravity*')
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
if rate > 0.8919:
    result.to_csv('result.csv', index=False)
    joblib.dump(clf, 'feature.pkl')
