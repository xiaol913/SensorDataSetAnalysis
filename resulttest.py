# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

raw = pd.read_csv('all_test_data.csv')
res = pd.read_csv('result.csv')

# print raw

# print res

# print res['Activity'][0]
count = 0
for i in range(0,len(raw)):
    if raw['Activity'][i] == res['Activity'][i]:
        count += 1
#
per = float(count)/float(len(raw))
#
print per
