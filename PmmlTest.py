from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.externals import joblib

clf = joblib.load('feature.pkl')

pipeline = PMMLPipeline([
    ("result", clf)
])

sklearn2pmml(pipeline, "result.pmml")
