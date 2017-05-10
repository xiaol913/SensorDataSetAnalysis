import pandas as pd
import numpy as np
import math
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


# GET MAX AND MIN
def analyze_data(data_set, activity):
    min_max_data = pd.DataFrame()

    AX_max = []
    AX_min = []
    AY_max = []
    AY_min = []
    AZ_max = []
    AZ_min = []

    GX_max = []
    GX_min = []
    GY_max = []
    GY_min = []
    GZ_max = []
    GZ_min = []

    LX_max = []
    LX_min = []
    LY_max = []
    LY_min = []
    LZ_max = []
    LZ_min = []

    for i in range(0, 3125):
        new = pd.DataFrame()
        new['AccelerometerX'] = normalize_data(data_set[i*40:((i+1)*40)]['AccelerometerX'])
        new['AccelerometerY'] = normalize_data(data_set[i*40:((i+1)*40)]['AccelerometerY'])
        new['AccelerometerZ'] = normalize_data(data_set[i*40:((i+1)*40)]['AccelerometerZ'])
        new['GravityX'] = normalize_data(data_set[i*40:((i+1)*40)]['GravityX'])
        new['GravityY'] = normalize_data(data_set[i*40:((i+1)*40)]['GravityY'])
        new['GravityZ'] = normalize_data(data_set[i*40:((i+1)*40)]['GravityZ'])
        new['LinearX'] = normalize_data(data_set[i*40:((i+1)*40)]['LinearX'])
        new['LinearY'] = normalize_data(data_set[i*40:((i+1)*40)]['LinearY'])
        new['LinearZ'] = normalize_data(data_set[i*40:((i+1)*40)]['LinearZ'])
        if math.isnan(new['AccelerometerX'].max()) | math.isnan(new['AccelerometerY'].max()) | math.isnan(new['AccelerometerZ'].max()) | math.isnan(new['GravityX'].max()) | math.isnan(new['GravityY'].max()) | math.isnan(new['GravityZ'].max()) | math.isnan(new['LinearX'].max()) | math.isnan(new['LinearY'].max()) | math.isnan(new['LinearZ'].max()):
            continue

        AX_max.append(new['AccelerometerX'].max())
        AX_min.append(new['AccelerometerX'].min())
        AY_max.append(new['AccelerometerY'].max())
        AY_min.append(new['AccelerometerY'].min())
        AZ_max.append(new['AccelerometerZ'].max())
        AZ_min.append(new['AccelerometerZ'].min())

        GX_max.append(new['GravityX'].max())
        GX_min.append(new['GravityX'].min())
        GY_max.append(new['GravityY'].max())
        GY_min.append(new['GravityY'].min())
        GZ_max.append(new['GravityZ'].max())
        GZ_min.append(new['GravityZ'].min())

        LX_max.append(new['LinearX'].max())
        LX_min.append(new['LinearX'].min())
        LY_max.append(new['LinearY'].max())
        LY_min.append(new['LinearY'].min())
        LZ_max.append(new['LinearZ'].max())
        LZ_min.append(new['LinearZ'].min())

    min_max_data['AX_max'] = AX_max
    min_max_data['AX_min'] = AX_min
    min_max_data['AY_max'] = AY_max
    min_max_data['AY_min'] = AY_min
    min_max_data['AZ_max'] = AZ_max
    min_max_data['AZ_min'] = AZ_min

    min_max_data['GX_max'] = GX_max
    min_max_data['GX_min'] = GX_min
    min_max_data['GY_max'] = GY_max
    min_max_data['GY_min'] = GY_min
    min_max_data['GZ_max'] = GZ_max
    min_max_data['GZ_min'] = GZ_min

    min_max_data['LX_max'] = LX_max
    min_max_data['LX_min'] = LX_min
    min_max_data['LY_max'] = LY_max
    min_max_data['LY_min'] = LY_min
    min_max_data['LZ_max'] = LZ_max
    min_max_data['LZ_min'] = LZ_min
    min_max_data['Activity'] = activity
    return min_max_data


raw_data = pd.read_csv('./data_set/raw_data.csv')
del raw_data['Unnamed: 0']
del raw_data['Timestamp']
# raw_data = analysis_data(raw_data)

# new = pd.DataFrame()
feet_data = analyze_data(raw_data[raw_data['Activity'] == 0][:125000], 0)
still_data = analyze_data(raw_data[raw_data['Activity'] == 1][:125000], 0)
vehicle_data = analyze_data(raw_data[raw_data['Activity'] == 2][:125000], 0)
# print(feet_data)

df = pd.DataFrame()
df = df.append(feet_data, ignore_index=True)
df = df.append(still_data, ignore_index=True)
df = df.append(vehicle_data, ignore_index=True)

print(df)
#
# X = new_data[['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
#               'GravityX', 'GravityY', 'GravityZ',
#               'LinearX', 'LinearY', 'LinearZ',
#               ]]
# y = new_data[['Activity']]
# train_pipeline = PMMLPipeline([
#     ("mapper", DataFrameMapper([
#         (['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
#               'GravityX', 'GravityY', 'GravityZ',
#               'LinearX', 'LinearY', 'LinearZ',
#           ],
#          [ContinuousDomain(), StandardScaler()])
#     ])),
#     ("pca", PCA(n_components=9)),
#     ("selector", SelectKBest(k=9)),
#     ("classifier", MLPClassifier())
# ])
#
# # predictions = cross_val_predict(train_pipeline, X, y.values.ravel(), cv=10)
# # print(metrics.accuracy_score(y.values.ravel(), predictions))
# # print(metrics.confusion_matrix(y.values.ravel(), predictions))
#
# # scores = cross_val_score(train_pipeline, X, y.values.ravel(), cv=5)
# # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
# predictions = train_pipeline.fit(X_train, y_train)
# score = predictions.score(X_test, y_test)
# print(score)
# y_pred = predictions.predict(X_test)
# print(metrics.confusion_matrix(y_test, y_pred))
#
# if score > .95:
#     sklearn2pmml(predictions, "./models/MLPClassifier_new.pmml")
#
# # neural_network.MLPClassifier          Accuracy: 0.82 (+/- 0.15)   0.950107868908
# # neural_network.MLPClassifier_new      Accuracy: 0.79 (+/- 0.39)   0.96944
