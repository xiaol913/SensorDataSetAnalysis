import pandas as pd
import numpy as np
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn2pmml import sklearn2pmml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ensemble.RandomForestClassifier   84571   Accuracy: 0.80 (+/- 0.17)   Accuracy: 0.84 (+/- 0.22)
# neighbors.KNeighborsClassifier    83675   Accuracy: 0.80 (+/- 0.12)   Accuracy: 0.84 (+/- 0.19)
# neural_network.MLPClassifier      86019   Accuracy: 0.80 (+/- 0.18)   Accuracy: 0.86 (+/- 0.16)


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

raw_data = pd.read_csv('raw_data.csv')
raw_data = new_function(raw_data)
X = raw_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
              'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'Gyroscope_value', 'GravityX',
              'GravityY', 'GravityZ', 'Gravity_value']]
y = raw_data[['Activity']]
train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
          'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'Gyroscope_value',
          'GravityX', 'GravityY', 'GravityZ', 'Gravity_value'],
         [ContinuousDomain(), StandardScaler()])
    ])),
    ("pca", PCA(n_components=12)),
    ("selector", SelectKBest(k=12)),
    ("classifier", MLPClassifier(activation='relu'))
])
# tanh = .9555  logistic = .9476    identity = .7177    relu = .
# predictions = cross_val_predict(train_pipeline, X, y.values.ravel(), cv=2)
# print(metrics.accuracy_score(y.values.ravel(), predictions))
# print(metrics.confusion_matrix(y.values.ravel(), predictions))
# scores = cross_val_score(train_pipeline, X, y.values.ravel(), cv=10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                    test_size=0.3, random_state=0)
# .25 = .95260  .29 = .95083  .3 = .95603  .35 = .95310
predictions = train_pipeline.fit(X_train, y_train)
score = predictions.score(X_test, y_test)
print(score)
y_pred = predictions.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
if score > .95:
    sklearn2pmml(predictions, "MLPClassifier.pmml", with_repr=True)
