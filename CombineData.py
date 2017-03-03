import glob
import pandas as pd


def read_data(files, activity):
    # column_names = ['accelerometerX', 'accelerometerY', 'accelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
    #                 'GravityX', 'GravityY', 'GravityZ', 'timestamp', 'Activity']
    column_names = ['accelerometerX', 'accelerometerY', 'accelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ', 'Activity']
    # column_names = ['accelerometerX', 'accelerometerY', 'accelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    # 'GravityX', 'GravityY', 'GravityZ']
    df = pd.DataFrame()
    # n = 0
    for i in range(0, len(files)/2):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        for j in range(0, len(df1)):
            # n += 20
            # df1['timestamp'][j] = n
            df1['Activity'][j] = activity

        df1['Activity'] = df1['Activity'].astype(int)
        df = df.append(df1, ignore_index=True)

    return df


InVehicle = glob.glob("./InVehicle/*")
Still = glob.glob("./Still/*")
Walking = glob.glob("./Walking/*")

InVehicleData = read_data(InVehicle, 0)
StillData = read_data(Still, 1)
WalkingData = read_data(Walking, 2)

# InVehicleData = read_data(InVehicle, 0)
# StillData = read_data(Still, 0)
# WalkingData = read_data(Walking, 0)

newData = pd.DataFrame()
newData = newData.append(InVehicleData)
newData = newData.append(StillData)
newData = newData.append(WalkingData)

newData.to_csv('all_data.csv')
# newData.to_csv('all_data_without_header_train.csv', header=None)
# newData.to_csv('all_data_without_header_test.csv', header=None)
# newData.to_csv('all_data_test.csv')
