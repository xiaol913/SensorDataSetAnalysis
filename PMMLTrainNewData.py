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
from scipy import stats
from sklearn import svm
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
    data_set['AccelerometerX'] = normalize_data(data_set['AccelerometerX'])
    data_set['AccelerometerY'] = normalize_data(data_set['AccelerometerY'])
    data_set['AccelerometerZ'] = normalize_data(data_set['AccelerometerZ'])
    data_set['GravityX'] = normalize_data(data_set['GravityX'])
    data_set['GravityY'] = normalize_data(data_set['GravityY'])
    data_set['GravityZ'] = normalize_data(data_set['GravityZ'])
    data_set['LinearX'] = normalize_data(data_set['LinearX'])
    data_set['LinearY'] = normalize_data(data_set['LinearY'])
    data_set['LinearZ'] = normalize_data(data_set['LinearZ'])
    return data_set


raw_data = pd.read_csv('./data_set/raw_data.csv')
del raw_data['Unnamed: 0']
del raw_data['Timestamp']
# raw_data = analysis_data(raw_data)

feet_data = raw_data[raw_data['Activity'] == 0][:125000]
still_data = raw_data[raw_data['Activity'] == 1][:125000]
vehicle_data = raw_data[raw_data['Activity'] == 2][:125000]

df = pd.DataFrame()
df = df.append(feet_data, ignore_index=True)
df = df.append(still_data, ignore_index=True)
df = df.append(vehicle_data, ignore_index=True)

new_data = analysis_data(df)

X = new_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
              'GravityX', 'GravityY', 'GravityZ',
              'LinearX', 'LinearY', 'LinearZ',
              ]]
y = new_data[['Activity']]
train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
              'GravityX', 'GravityY', 'GravityZ',
              'LinearX', 'LinearY', 'LinearZ',
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

if score > .95:
    sklearn2pmml(predictions, "./models/MLPClassifier_new.pmml")

# neural_network.MLPClassifier          Accuracy: 0.82 (+/- 0.15)   0.950107868908
# neural_network.MLPClassifier_new      Accuracy: 0.79 (+/- 0.39)   0.96944
