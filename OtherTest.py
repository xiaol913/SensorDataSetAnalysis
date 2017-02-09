import pandas as pd

# df = pd.read_csv('all_data_without_header_train.csv', header=None)
df = pd.read_csv('all_data_without_header_test.csv', header=None)
# df = pd.read_csv('all_data.csv')
# df.drop('0')
print df.dtypes
