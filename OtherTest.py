import pandas as pd
import numpy as np
from sklearn.externals import joblib


# normalize
def normalize(data_set):
    mu = np.mean(data_set, axis=0)
    sigma = np.std(data_set, axis=0)
    return (data_set - mu) / sigma


data_set = pd.read_csv('./data_set/raw_data.csv')

print(np.mean(data_set['AccelerometerX']))
print(np.mean(data_set['AccelerometerY']))
print(np.mean(data_set['AccelerometerZ']))
print(np.mean(data_set['GravityX']))
print(np.mean(data_set['GravityY']))
print(np.mean(data_set['GravityZ']))
print(np.mean(data_set['LinearX']))
print(np.mean(data_set['LinearY']))
print(np.mean(data_set['LinearZ']))
