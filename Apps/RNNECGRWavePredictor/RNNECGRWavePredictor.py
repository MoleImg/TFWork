'''
Todo:
RNN based ECG R peak prediction
Input:
Series of R peak postions
Output:
Probability distributions of R peak position in some specific time windows
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import RNNToolBox as rtb
import DataParser as dp
import ModelConductor as mc


def get_interval(time_data):
    interval_num = time_data.shape[0]-1
    time_interval = np.zeros(shape=[interval_num, 1])
    for i in range(interval_num):
        time_interval[i] = time_data[i+1] - time_data[i]

    return time_interval


def calc_prob(predict_data, time_range):
    '''
    calculate the probability distribution of the specific time range
    :param predict_data: the data predicted by RNN model
    :param time_range: the time window range of the probability (numpy array)
    :return: independent probability distribution on the time_range, corresponding to the predicted data
    '''
    prob = (1-tf.nn.softmax(abs(time_range-predict_data))) / time_range.shape[0]
    return prob



#------hyper parameters-----#
IS_TRAINING = True # construct the new model/load the old model
IS_SAVE_MODEL = True # save model or not
IS_PLOT = True
DATA_PATH = '../Data/MIT_peak_data/'
CHECK_PATH = '../Apps/RNNECGRWavePredictor/model_checkpoint/'
SAVE_PATH = '../Apps/RNNECGRWavePredictor/model_checkpoint/params'

TRAIN_PROB = 0.9
TIME_STEP = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 1
TRAIN_BATCH_SIZE = 10
CELL_NUM = 64
CELL_LAYER_NUM = None

NN_TYPE = 'LSTM'
LR = 0.01
IS_LR_DECAY = False
TRAIN_EPOCH = 1000
GLOBAL_EPOCH = 0
PRED_TIME_RANGE = 325

#-----data import----#

raw_R_peak = dp.read_data_from_txt(DATA_PATH+'103Rpeak.txt')
print(raw_R_peak.shape)
# plt.plot(raw_R_peak)
# plt.show()
time_interval = get_interval(raw_R_peak)
# time_interval = (time_interval - np.mean(time_interval)) / np.std(time_interval)
# plt.plot(time_interval)
# plt.show()
print(time_interval.shape)
train_num = int(time_interval.shape[0]*TRAIN_PROB)
# train_num = 20
test_num = int(time_interval.shape[0]*(1-TRAIN_PROB))
train_time_interval = time_interval[:train_num]
# standardization
train_time_interval_std = (train_time_interval - np.mean(train_time_interval)) / np.std(train_time_interval)
test_time_interval = time_interval[train_num-TIME_STEP:]
# standardization
test_time_interval_std = (test_time_interval - np.mean(test_time_interval)) / np.std(test_time_interval)

train_x, train_y = dp.data_create(train_time_interval_std, TIME_STEP)
test_x, test_y = dp.data_create(test_time_interval_std, TIME_STEP)
# print(data_x.shape)
# print(data_y.shape)

# print('TX:',train_x)
# print('TY:',train_y)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# print('T1:',train_x[-1,:,0])
# print('T2:',train_y[-1,:,0])
# print('T3:',test_x[0,:,0])
# print('T4:',test_y[0,:,0])


#--------model graph dictionary----#
'''
single layer RNN
'''
model_dict = mc.model_graph(INPUT_SIZE, OUTPUT_SIZE, CELL_NUM, TIME_STEP,
                            NN_TYPE, cell_layer_num= CELL_LAYER_NUM, learning_rate=LR)

if IS_TRAINING:

    #---------training--------#
    mc.train_model(train_x, train_y, model_dict, TRAIN_EPOCH, TRAIN_BATCH_SIZE,learning_rate=LR,
                   is_learn_decay=IS_LR_DECAY, is_save=IS_SAVE_MODEL, save_path=SAVE_PATH, checkpoint_path=CHECK_PATH)

else:
    # test training set
    print('[INFO] Testing training set')
    train_pred = mc.test_model(train_x, train_y, model_dict, checkpoint_path=CHECK_PATH)
    # test testing set
    print('[INFO] Testing testing set')
    test_pred = mc.test_model(test_x, test_y, model_dict, checkpoint_path=CHECK_PATH)

    # recover data: destandardization
    train_pred = train_pred[:,0,0]*np.std(train_time_interval) + np.mean(train_time_interval)
    train_label = train_y[:,-1,0]*np.std(train_time_interval) + np.mean(train_time_interval)
    test_pred = test_pred[:,0,0]*np.std(test_time_interval) + np.mean(test_time_interval)
    test_label = test_y[:,-1,0]*np.std(test_time_interval) + np.mean(test_time_interval)
    # print('#############')
    # print(test_pred.shape)
    # print(test.shape)
    if IS_PLOT:
        plt.figure(1)
        plt.plot(train_pred,'r-')
        plt.plot(train_label, 'b-')
        plt.xlabel('beats num.')
        plt.ylabel('intervals')
        plt.title('Training results')
        plt.legend(('Predictions','Labels'))
        plt.figure(2)
        plt.plot(test_pred,'r-')
        plt.plot(test_label, 'b-')
        plt.xlabel('beats num.')
        plt.ylabel('intervals')
        plt.title('Testing results')
        plt.legend(('Predictions','Labels'))
        plt.show()


    # print('Probability distribution...')
    # prob_mat = np.zeros(shape=(pred_y.shape[0], len(PRED_TIME_RANGE)))
    # for i in range(pred_y.shape[0]):
    #     prob_mat[i,:] = sess.run(calc_prob(pred_y[i,0,0], np.array(PRED_TIME_RANGE)))








