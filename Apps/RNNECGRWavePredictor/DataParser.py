'''
Todo:
ECG data import and realign
'''
import numpy as np

def read_data_from_txt(file_name):
    with open(file_name, 'r') as f:
        data = []
        while True:
            lines = f.readline()
            if not lines:
                break
            lines_list = lines.split(r',')

            try:
                for i in range(len(lines_list)):
                    data.append(float(lines_list[i]))
            except ValueError:
                print(lines_list[i])
    return np.array(data).reshape((len(data),1))

def data_create(data, time_step=5):
    data_x, data_y = [], []
    for i in range(data.shape[0] - time_step):
        x = data[i:(i+time_step),:]
        y = data[(i+1):(i+time_step+1),:]
        data_x.append(x.tolist())
        data_y.append(y.tolist())
    return np.array(data_x), np.array(data_y)


def get_test_data(input_data, pred_data):
    idx = 1
    while(idx<input_data.shape[1]):
        input_data[:,idx-1,:] = input_data[:,idx,:]
        idx += 1
    input_data[:,-1,:] = pred_data

    return input_data
