import pandas as pd
import numpy as np
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn2pmml import sklearn2pmml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# linear_model  RidgeClassifier(Normalizer=838)
# LogisticRegression(Normalizer=847)    LogisticRegressionCV(Normalizer=845)
# RidgeClassifierCV(Normalizer=848)   SGDClassifier(Normalizer=849)


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
    return data_set


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
result_data = pd.read_csv('result_data.csv')
train_data = new_function(train_data)
test_data = new_function(test_data)
train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (["AccelerometerX", "AccelerometerY", "AccelerometerZ", "Accelerometer_value"],
         [ContinuousDomain(), StandardScaler()])
    ])),
    ("pca", PCA(n_components=4)),
    ("selector", SelectKBest(k=4)),
    ("classifier", SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, l1_ratio=0.15,
                                 fit_intercept=True, n_iter=5, shuffle=True, verbose=0,
                                 epsilon=0.1, n_jobs=1, random_state=None,
                                 learning_rate='optimal', eta0=0.0, power_t=1.0,
                                 class_weight=None, warm_start=False, average=False))
])
train_pipeline = train_pipeline.fit(train_data, train_data['Activity'])
test = test_data.filter(regex='Accelerometer')

predictions = train_pipeline.predict(test)
result = pd.DataFrame({'Activity': predictions.astype(np.int32)})

# compute the matching rate
count = 0
for i in range(0, len(result)):
    if result['Activity'][i] == result_data['Activity'][i]:
        count += 1

rate = count / len(result)
print(rate)

if rate > 0.8574:
    result.to_csv('result.csv', index=False)
    # joblib.dump(train_pipeline, "newPKL.pkl.z", compress=9)
    # save as PMML
    sklearn2pmml(train_pipeline, "SGDClassifier.pmml", with_repr=True)
