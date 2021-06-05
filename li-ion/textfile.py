import numpy as np
import os
import datetime
import pickle

# Run this script if text file is not available

if not os.path.exists('charge_cycle_input.txt'):
    df = pickle.load(open('raw_data.pkl', 'rb'))
    with open('charge_cycle_input.txt', 'w') as f:
        index = np.random.randint(0, df.shape[0])
        for i, j in zip(df.loc[index].index,df.loc[index]):
            f.write(i + ',')
            if (isinstance(j, str)):
                f.write(j)
            elif (isinstance(j, datetime.datetime)):
                f.write(str(j))
            elif (isinstance(j, np.ndarray)):
                for k in j:
                    f.write(str(k)+',')
            elif (isinstance(j, np.int64)):
                f.write(str(j))
            else:
                f.write(str(j))

            f.write('\n')
    f.close()
