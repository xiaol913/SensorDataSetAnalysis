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


raw_data = pd.read_csv('./data_set/raw_data.csv')
del raw_data['Unnamed: 0']
del raw_data['Timestamp']
new_data = analysis_data(raw_data)
# print(new_data)
X = new_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
              'GravityX', 'GravityY', 'GravityZ',
              'LinearX', 'LinearY', 'LinearZ'
              ]]
y = new_data[['Activity']]
train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
          'GravityX', 'GravityY', 'GravityZ',
          'LinearX', 'LinearY', 'LinearZ'
          ],
         [ContinuousDomain(), StandardScaler()])
    ])),
    ("pca", PCA(n_components=9)),
    ("selector", SelectKBest(k=9)),
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

# if score > .948:
#     sklearn2pmml(predictions, "./models/MLPClassifier.pmml")

# neighbors.KNeighborsClassifier        Accuracy: 0.74 (+/- 0.15)   0.931170064721
# ensemble.RandomForestClassifier       Accuracy: 0.73 (+/- 0.16)   0.928357373742
# ensemble.BaggingClassifier            Accuracy: 0.73 (+/- 0.16)   0.927665245595
# ensemble.GradientBoostingClassifier   Accuracy: 0.68 (+/- 0.23)   0.911098348464
# tree.ExtraTreeClassifier              Accuracy: 0.72 (+/- 0.17)   0.910141149964
# tree.DecisionTreeClassifier           Accuracy: 0.72 (+/- 0.17)   0.911834655004
# neural_network.MLPClassifier          Accuracy: 0.82 (+/- 0.15)   0.950107868908
