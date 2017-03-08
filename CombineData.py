import glob

import pandas as pd


def read_data_train(files, activity):
    column_names = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ', 'Timestamp', 'Activity']
    df = pd.DataFrame()
    n = 0
    for i in range(0, len(files) / 2):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        for j in range(0, len(df1)):
            n += 20
            df1['Timestamp'][j] = n
            df1['Activity'][j] = activity

        # transform type to int
        df1['Activity'] = df1['Activity'].astype(int)
        df = df.append(df1, ignore_index=True)

    return df


def read_data_test_no_result(files):
    column_names = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ']
    df = pd.DataFrame()
    for i in range(len(files) / 2, len(files)):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        df = df.append(df1, ignore_index=True)

    return df


def read_data_test(files, activity):
    column_names = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ', 'Activity']
    df = pd.DataFrame()
    for i in range(len(files) / 2, len(files)):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        for j in range(0, len(df1)):
            df1['Activity'][j] = activity

        # transform type to int
        df1['Activity'] = df1['Activity'].astype(int)
        df = df.append(df1, ignore_index=True)

    return df


InVehicle = glob.glob("./InVehicle/*")
Still = glob.glob("./Still/*")
Walking = glob.glob("./Walking/*")

# combine train data
train_data = pd.DataFrame()
train_vd = read_data_train(InVehicle, 0)
train_sd = read_data_train(Still, 1)
train_wd = read_data_train(Walking, 2)
train_data = train_data.append(train_vd)
train_data = train_data.append(train_sd)
train_data = train_data.append(train_wd)
train_data.to_csv('train_data.csv')

# combine test data
test_data = pd.DataFrame()
test_vd = read_data_test_no_result(InVehicle)
test_sd = read_data_test_no_result(Still)
test_wd = read_data_test_no_result(Walking)
test_data = test_data.append(test_vd)
test_data = test_data.append(test_sd)
test_data = test_data.append(test_wd)
train_data.to_csv('test_data.csv')

# combine result data
result_data = pd.DataFrame()
result_vd = read_data_test(InVehicle, 0)
result_sd = read_data_test(Still, 1)
result_wd = read_data_test(Walking, 2)
result_data = result_data.append(result_vd)
result_data = result_data.append(result_sd)
result_data = result_data.append(result_wd)
train_data.to_csv('result_data.csv')
