import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare(datasets, test_scale, timesteps, scale_min=0, scale_max=1):
    """
    :returns: list(training_sets), list(test_sets), scaler
    """
    def prep(dset:pd.DataFrame):
        columns_to_drop = ['rotational_vector_x',
                           'rotational_vector_y',
                           'rotational_vector_z',
                           'gravity_x',
                           'gravity_y',
                           'gravity_z',
                           'magnetic_field_x',
                           'magnetic_field_y',
                           'magnetic_field_z',
                           'gyroscope_x',
                           'gyroscope_y',
                           'gyroscope_z',
                           'angular_velocity_x',
                           'angular_velocity_y',
                           'angular_velocity_z',
                           'id',
                           'maxerror',
                           'Vin',
                           'Adin',
                           'changecurr',
                           'rollAngle',
                           'pitchAngle',
                           'temperature']
        for name in columns_to_drop:
            try:
                dset.drop(name, axis=1, inplace=True)
            except ValueError:
                continue
        dset.replace(13371337, np.NaN, inplace=True)
        dset.fillna(method='ffill', inplace=True)
        dset.fillna(method='bfill', inplace=True)
        return dset

    def percent_complete(num_rows):
        return [i / num_rows for i in range(num_rows)]

    def rate_terrain(name, num_rows):
        if 'roof' in name:
            return [0 for i in range(num_rows)]
        elif 'field' in name:
            return [1 for i in range(num_rows)]
        
    def calculate_movement_gradient(df, num_rows):
        gradient_list = []
        for i in range(num_rows):
            gradient = float((df['linear_acceleration_x'][i]**2 + df['linear_acceleration_y'][i]**2 + df['linear_acceleration_z'][i]**2)**(1/2))
            gradient_list.append(gradient)
        return tuple(gradient_list)

    def coulomb_sum(df, num_rows):
        total = 0
        last_ts = 0
        sum_list = []
        for row in range(num_rows):
            current_time = df['time'][row]
            if current_time == last_ts:
                sum_list.append(total)
                continue
            last_ts = current_time
            current_val = df['current'][row] / 1000
            total += current_val
            sum_list.append(total)
        return sum_list

    def voltage_mavg(df, num_rows):
        mavg_num = 50
        new_df = df.rolling(mavg_num)['voltage'].mean()
        quick_total = 0
        for i in range(mavg_num):
            quick_total = quick_total + df['voltage'][i]
            new_df[i] = quick_total / (i+1)
        return new_df.values.tolist()

    def prep_and_add(name, dataset):
        final_set = {}
        num_rows = dataset.shape[0]
        dataset = prep(dataset)
        try:
            final_set['gradient'] = calculate_movement_gradient(dataset, num_rows)
        except KeyError:
            return False
        
        # percent completion of the trip
        final_set['percent_complete'] = percent_complete(num_rows)
        # terrain
        final_set['terrain'] = rate_terrain(name, num_rows)
        # coulomb sum
        final_set['coulombs_consumed'] = coulomb_sum(dataset, num_rows)
        # voltage mavg
        final_set['voltage_mavg'] = voltage_mavg(dataset, num_rows)
        # state of charge
        final_set['soc'] = dataset['abssoc'].tolist()
        # target_soc
        final_set['target_soc'] = [dataset['abssoc'][num_rows - 1] for i in range(num_rows)]
        return final_set
    
    def to_x_and_y(dset):
        if type(dset) == list:
            return False
        x_tr_set = []
        y_tr_set = [dset[i][-1] for i in range(timesteps, dset.shape[0])]
        
        for x in range(dset.shape[1]-1):
            x_tr = []
            for i in range(timesteps, dset.shape[0]):
                x_tr.append(dset[i-timesteps:i, 0])
            x_tr_set.append(x_tr)
        x_tr_set = np.array(x_tr_set)
        if x_tr_set.shape[1] < 3:
            return False
        y_tr_set = np.array(y_tr_set)

        return [x_tr_set, y_tr_set]
        

    dataframes = []
    if type(datasets) == str:
        datasets = [datasets]

    max_values = {'coulombs_consumed':0,
                  'gradient':0,
                  'percent_complete':0,
                  'soc':0,
                  'terrain':0,
                  'voltage_mavg':0,
                  'target_soc':0}
    min_values = {'coulombs_consumed':0,
                  'gradient':0,
                  'percent_complete':0,
                  'soc':0,
                  'terrain':0,
                  'voltage_mavg':0,
                  'target_soc':0}
    column_order = ['coulombs_consumed',
                    'gradient',
                    'percent_complete',
                    'soc',
                    'terrain',
                    'voltage_mavg',
                    'target_soc']
    for dset in datasets:
        data_dict = prep_and_add(dset, pd.read_csv(dset))
        if not data_dict:
            print(dset.split('/')[-1], 'is not a usable dataset')
            continue
        frame = pd.DataFrame.from_dict(data_dict)
        fmax = frame.max()
        fmin = frame.min()
        for name in column_order:
            if fmax[name] > max_values[name]:
                max_values[name] = fmax[name]
            if fmin[name] < min_values[name]:
                min_values[name] = fmin[name]
        frame = frame[column_order]
        dataframes.append(frame)
        print(dset.split('/')[-1], 'will be used')
    

    # inputs: dataset, test_scale, scale_min, scale_max
    training_sets = []
    test_sets = []
    print('scaling datasets')
    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
    for df in dataframes:
        #size = df.shape[0]
        #training_set = df.values
        #training_set = astype('float32')
        # prepare the scaler against all data
        scaler.fit(df.values)
        
    for df in dataframes:
        size = df.shape[0]
        test_set = scaler.transform(df.iloc[::10, :].values)
        training_set = scaler.transform(df.values)
        np.delete(training_set, list(range(0, size, 10)), axis=0)
       
        training_set = to_x_and_y(training_set)
        if not training_set: continue
        training_set[0] = np.swapaxes(training_set[0], 0, 2)
        training_set[0] = np.swapaxes(training_set[0], 0, 1)
        test_set = to_x_and_y(test_set)
        if not test_set: continue
        test_set[0] = np.swapaxes(test_set[0], 0, 2)
        test_set[0] = np.swapaxes(test_set[0], 0, 1)
        
        training_sets.append(training_set)
        test_sets.append(test_set)

    print('dataset scaling complete')
        
    X_train = np.concatenate([i[0] for i in training_sets],axis=0)
    Y_train = np.concatenate([i[1] for i in training_sets],axis=0)
    X_test = np.concatenate([i[0] for i in test_sets],axis=0)
    Y_test = np.concatenate([i[1] for i in test_sets],axis=0)
  
    return X_train, Y_train, X_test, Y_test, scaler

