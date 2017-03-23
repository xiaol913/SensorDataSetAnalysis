import glob

import pandas as pd


def read_raw_data(files):
    column_names = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ', 'OldTimestamp', 'Activity', 'Timestamp']
    df = pd.DataFrame()
    n = 0
    for i in range(0, len(files)):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        for j in range(0, len(df1)):
            n += 20.0
            df1.loc[j, 'Timestamp'] = n

        df = df.append(df1, ignore_index=True)

    del df['GyroscopeX']
    del df['GyroscopeY']
    del df['GyroscopeZ']
    del df['OldTimestamp']
    return df


def add_linear(data_set):
    data_set['LinearX'] = data_set['AccelerometerX'] - data_set['GravityX']
    data_set['LinearY'] = data_set['AccelerometerY'] - data_set['GravityY']
    data_set['LinearZ'] = data_set['AccelerometerZ'] - data_set['GravityZ']
    return data_set


InVehicle = glob.glob("./data_set/InVehicle/*")
Still = glob.glob("./data_set/Still/*")
OnFeet = glob.glob("./data_set/OnFeet/*")

# combine raw data
raw_data = pd.DataFrame()

raw_vd = read_raw_data(InVehicle)
raw_sd = read_raw_data(Still)
raw_wd = read_raw_data(OnFeet)

raw_data = raw_data.append(raw_vd)
raw_data = raw_data.append(raw_sd)
raw_data = raw_data.append(raw_wd)

raw_data = add_linear(raw_data)

# del raw_data['Unnamed: 0']
# del data_set['Timestamp']
feet_data = raw_data[raw_data['Activity'] == 0][:125000]
still_data = raw_data[raw_data['Activity'] == 1][:125000]
vehicle_data = raw_data[raw_data['Activity'] == 2][:125000]

df = pd.DataFrame()
df = df.append(feet_data, ignore_index=True)
df = df.append(still_data, ignore_index=True)
df = df.append(vehicle_data, ignore_index=True)

df.to_csv("./data_set/raw_data.csv")
