import numpy as np


def avgoutput(lst):
    '''get average output'''
    sample = []
    rem = len(lst)//10
    ini = 0
    fin = rem
    if len(lst) > 10:
        for j in range(10):
            sample.append(np.nanmean(lst[ini:fin]))
            ini = ini + rem 
            fin = fin + rem       
    return sample     


def preprocess(file):
    padding_length = 3900 # obtained from training data
    voltage, current, temperature = np.array([]), np.array([]), np.array([])
    with open(file, 'r') as f:
        x = f.readlines()
        temp = [float(x[1].split(",")[1])]
        for i in x[3: len(x)-1]:
            splitted_string = i.replace('\n', '').split(',')
            if splitted_string[0] == "voltage_measured":
                for j in (splitted_string[1: len(splitted_string)-1]):
                    voltage = np.append(voltage, float(j))
            if splitted_string[0] == "temperature_measured":
                for j in (splitted_string[1: len(splitted_string)-1]):
                    temperature = np.append(temperature, float(j))
            if splitted_string[0] == "current_charge":
                for j in (splitted_string[1: len(splitted_string)-1]):
                    current = np.append(current, float(j))             
        f.close()
        voltage = avgoutput(np.append((padding_length - len(voltage)) * [0], voltage))
        current = avgoutput(np.append((padding_length - len(current)) * [0], current))
        temperature = avgoutput(np.append((padding_length - len(temperature)) * [0], temperature))
        temp.extend(voltage)
        temp.extend(temperature)
        temp.extend(current)
    return np.asarray(temp).reshape(1, -1)
