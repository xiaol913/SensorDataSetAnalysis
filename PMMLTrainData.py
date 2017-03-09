import pandas as pd
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer
from sklearn.linear_model import RidgeClassifier
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml
import numpy as np

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
result_data = pd.read_csv('result_data.csv')

train_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (["AccelerometerX", "AccelerometerY", "AccelerometerZ"], [ContinuousDomain(), Imputer()])
    ])),
    ("classifier", RidgeClassifier(alpha=0.001, fit_intercept=True, normalize=True,
                                   copy_X=True, max_iter=1000, tol=0.000001,
                                   class_weight=None, solver='auto', random_state=None))
])
train_pipeline.fit(train_data, train_data['Activity'])
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

if rate > 0.86:
    result.to_csv('result.csv', index=False)
    # save as PMML
    sklearn2pmml(train_pipeline, "RidgeClassifier.pmml", with_repr=True)
