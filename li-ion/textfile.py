import numpy as np
import os
import datetime

# Run this script if text file is not available

if not os.path.exists('charge_cycle_input.txt'):
    with open('charge_cycle_input.txt', 'w') as f:
        index = np.random.randint(0, ch[14].shape[0])
        for i, j in zip(ch[14].loc[index].index,ch[14].loc[index]):
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
