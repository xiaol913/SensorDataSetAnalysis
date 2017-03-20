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

# neighbors.KNeighborsClassifier    Accuracy: 0.80 (+/- 0.12)   Accuracy: 0.84 (+/- 0.19)   0.975561985966
# ensemble.RandomForestClassifier   Accuracy: 0.80 (+/- 0.17)   Accuracy: 0.84 (+/- 0.22)   0.974509067615
# ensemble.GradientBoostingClassifier   Accuracy: 0.78 (+/- 0.19)                           0.944990538461
# ensemble.BaggingClassifier        Accuracy: 0.78 (+/- 0.13)   Accuracy: 0.85 (+/- 0.18)   0.972528403025
# tree.ExtraTreeClassifier          Accuracy: 0.76 (+/- 0.13)   Accuracy: 0.81 (+/- 0.23)   0.960695956941
# tree.DecisionTreeClassifier       Accuracy: 0.76 (+/- 0.13)   Accuracy: 0.83 (+/- 0.20)   0.95448152975
# neural_network.MLPClassifier      Accuracy: 0.80 (+/- 0.18)   Accuracy: 0.86 (+/- 0.16)   0.9495777282


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
    # data_set['GyroscopeX'] = abs(data_set['GyroscopeX'])
    # data_set['GyroscopeY'] = abs(data_set['GyroscopeY'])
    # data_set['GyroscopeZ'] = abs(data_set['GyroscopeZ'])
    # data_set['Gyroscope_value'] = data_set['GyroscopeX'] ** 2 + \
    #                                   data_set['GyroscopeY'] ** 2 + \
    #                                   data_set['GyroscopeZ'] ** 2
    data_set['GravityX'] = abs(data_set['GravityX'])
    data_set['GravityY'] = abs(data_set['GravityY'])
    data_set['GravityZ'] = abs(data_set['GravityZ'])
    data_set['Gravity_value'] = data_set['GravityX'] ** 2 + \
                                      data_set['GravityY'] ** 2 + \
                                      data_set['GravityZ'] ** 2
    data_set['LinearX'] = abs(data_set['AccelerometerX'] - data_set['GravityX'])
    data_set['LinearY'] = abs(data_set['AccelerometerY'] - data_set['GravityY'])
    data_set['LinearZ'] = abs(data_set['AccelerometerZ'] - data_set['GravityZ'])
    data_set['Linear_value'] = data_set['LinearX'] ** 2 + \
                                data_set['LinearY'] ** 2 + \
                                data_set['LinearZ'] ** 2
    return data_set

raw_data = pd.read_csv('raw_data.csv')
raw_data = new_function(raw_data)
X = raw_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
              # 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'Gyroscope_value',
              'GravityX', 'GravityY', 'GravityZ', 'Gravity_value',
              'LinearX', 'LinearY', 'LinearZ', 'Linear_value'
              ]]
# X = raw_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
#               # 'Gyroscope_value',
#               # 'Gravity_value'
#               ]]
y = raw_data[['Activity']]
train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'Accelerometer_value',
          # 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'Gyroscope_value',
          'GravityX', 'GravityY', 'GravityZ', 'Gravity_value',
          'LinearX', 'LinearY', 'LinearZ', 'Linear_value'
          ],
         [ContinuousDomain(), StandardScaler()])
    ])),
    ("pca", PCA(n_components=12)),
    ("selector", SelectKBest(k=12)),
    ("classifier", MLPClassifier())
])
# predictions = cross_val_predict(train_pipeline, X, y.values.ravel(), cv=10)
# print(metrics.accuracy_score(y.values.ravel(), predictions))
# print(metrics.confusion_matrix(y.values.ravel(), predictions))

# scores = cross_val_score(train_pipeline, X, y.values.ravel(), cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
predictions = train_pipeline.fit(X_train, y_train)
score = predictions.score(X_test, y_test)
print(score)
y_pred = predictions.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))

if score > .941:
    sklearn2pmml(predictions, "./models/MLPClassifier.pmml")

# neighbors.KNeighborsClassifier        Accuracy: 0.74 (+/- 0.15)   0.931170064721
# ensemble.RandomForestClassifier       Accuracy: 0.73 (+/- 0.16)   0.928357373742
# ensemble.BaggingClassifier            Accuracy: 0.73 (+/- 0.16)   0.927665245595
# ensemble.GradientBoostingClassifier   Accuracy: 0.68 (+/- 0.23)   0.911098348464
# tree.ExtraTreeClassifier              Accuracy: 0.72 (+/- 0.17)   0.910141149964
# tree.DecisionTreeClassifier           Accuracy: 0.72 (+/- 0.17)   0.911834655004
# neural_network.MLPClassifier          Accuracy: 0.75 (+/- 0.14)   0.918638127425
