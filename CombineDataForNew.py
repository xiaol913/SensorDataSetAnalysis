import glob

import pandas as pd


def read_raw_data(files):
    column_names = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                    'GravityX', 'GravityY', 'GravityZ', 'Timestamp', 'Activity']
    df = pd.DataFrame()
    for i in range(0, len(files)):
        df1 = pd.read_csv(files[i], header=None, names=column_names)
        df = df.append(df1, ignore_index=True)

    return df


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

raw_data.to_csv("raw_data.csv")
