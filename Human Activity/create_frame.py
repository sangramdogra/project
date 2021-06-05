import pickle 
import numpy as np
import os
import pandas as pd

if not os.path.exists('frame.pkl'):
    # Data directory
    DATADIR = 'Dataset'

    ACTIVITIES = {
        0: 'WALKING',
        1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING',
        4: 'STANDING',
        5: 'LAYING',
    }

    SIGNALS = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z"
    ]

    # Utility function to read the data from csv file
    def _read_csv(filename):
        return pd.read_csv(filename, delim_whitespace=True, header=None)

    # Utility function to load the load
    def load_signals(subset):
        signals_data = []

        for signal in SIGNALS:
            filename = f'Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'
            signals_data.append(
                _read_csv(filename).to_numpy()
            ) 

        # Transpose is used to change the dimensionality of the output,
        # aggregating the signals by combination of sample/timestep.
        # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
        return np.transpose(signals_data, (1, 2, 0))

    def load_y(subset):
        """
        The objective that we are trying to predict is a integer, from 1 to 6,
        that represents a human activity.
        """
        filename = f'Dataset/{subset}/y_{subset}.txt'
        y = _read_csv(filename)[0]

        return pd.get_dummies(y).to_numpy()

    def load_data():
        """
        Obtain the dataset from multiple files.
        Returns: X_train, X_test, y_train, y_test
        """
        X_test = load_signals('test')
        y_test = load_y('test')

        return  X_test, y_test

    with open('frame.pkl', 'wb') as f:
        x, y = load_data()
        index = np.random.randint(0,x.shape[0])
        pickle.dump(x[index].reshape(1, 128, -1), f)
        f.close()
