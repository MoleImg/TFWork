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
import pandas as pd
import RNNToolBox as rtb
import DataParser as dp
import ModelConductor as mc

def get_batch_data(data_x, data_y, batch_size):
    global start, end
    batch_x, batch_y = data_x[start:end,:,:], data_y[start:end,:,:]
    start += batch_size
    end = start + batch_size
    return batch_x, batch_y


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
IS_MODELLING = False # construct the new model/load the old model

TRAIN_PROB = 0.9
TIME_STEP = 5
INPUT_SIZE = 1
OUTPUT_SIZE = 1
TRAIN_BATCH_SIZE = 10
CELL_NUM = 64
CELL_LAYER_NUM = 4

NN_TYPE = 'BID'
LR = 0.01
IS_LR_DECAY = False
TRAIN_EPOCH = 1000
GLOBAL_EPOCH = 0

ROOT_PATH = '../Data/MIT_peak_data/'
PRED_TIME_RANGE = 325

#-----data import----#
# np.random.seed(1)
# R_peak = np.linspace(1,100,500)[:, np.newaxis]
# R_peak = R_peak**2
# noise = np.random.normal(0, 0.5, size=R_peak.shape)
# R_peak += noise

raw_R_peak = dp.read_data_from_txt(ROOT_PATH+'100Rpeak.txt')
print(raw_R_peak.shape)
# plt.plot(raw_R_peak)
# plt.show()
time_interval = get_interval(raw_R_peak)
time_interval = (time_interval - np.mean(time_interval)) / np.std(time_interval)
# plt.plot(time_interval)
# plt.show()
print(time_interval.shape)
train_num = int(time_interval.shape[0]*TRAIN_PROB)
# train_num = 20
test_num = int(time_interval.shape[0]*(1-TRAIN_PROB))
data_x, data_y = dp.data_create(time_interval, TIME_STEP)
# print(data_x.shape)
# print(data_y.shape)
train_x, train_y = data_x[:train_num,:,:], data_y[:train_num,:,:]
test_x, test_y = data_x[train_num:,:,:], data_y[train_num:,:,:]
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

# # ---------------data import------------#
# raw_df = pd.read_csv('../Data/httpRequestData.csv')
# # plt.plot(raw_df)
# # plt.show()
# raw_data = raw_df.values
# raw_data = raw_data.astype('float32')
#
# # normalization
# raw_data_norm = (raw_data - np.mean(raw_data)) / np.std(raw_data)
# print('Shape of normalized raw data:', raw_data_norm.shape)
# # plt.plot(raw_data_norm)
# # plt.show()
#
# # -------------hyper parameters----------#
# TIME_STEP = 3
# TRAIN_BATCH_SIZE = 5
# INPUT_SIZE = 1
# OUTPUT_SIZE = 1
# CELL_NUM = 64
#
# LR = 0.01
# TRAIN_EPOCH = 2000
#
# # -------------Train & Test data---------#
# data_x, data_y = data_create(raw_data_norm, TIME_STEP)
# print('Shape of raw data:', raw_data_norm.shape)
# print('Shape of data x:', data_x.shape)
# print('Shaoe of data y:', data_y.shape)
# # train_data = raw_data_norm[:int(0.8*raw_data_norm.shape[0]),:]
# # train_x, train_y = create_dataset(train_data, TIME_STEP)
# train_x, train_y = data_x[:int(0.8*raw_data_norm.shape[0]),:,:], data_y[:int(0.8*raw_data_norm.shape[0]),:,:]
# print('Shape of training input:', train_x.shape)
# print('Shape of training label:', train_y.shape)
# # print(train_x)
# # test_data = raw_data_norm[int(0.8*raw_data_norm.shape[0]):,:]
# # test_x, test_y = create_dataset(test_data, TIME_STEP)
# test_x, test_y = data_x[int(0.8*raw_data_norm.shape[0]):,:,:], data_y[int(0.8*raw_data_norm.shape[0]):,:,:]
# print('Shape of testing input:', test_x.shape)
# print('Shape of testing label:', test_y.shape)
#
# print('T1:',train_x[-1,:,0])
# print('T2:',train_y[-1,:,0])
# print('T3:',test_x[0,:,0])
# print('T4:',test_y[0,:,0])

if IS_MODELLING:
    #--------model graph dictionary----#
    '''
    single layer RNN
    '''
    model_dict = mc.model_graph(NN_TYPE, INPUT_SIZE, OUTPUT_SIZE, CELL_NUM, TIME_STEP,
                                TRAIN_EPOCH, learning_rate=LR, is_learn_decay=False)

    #--------model graph-----#
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
        input_y = tf.placeholder(tf.float32, [None, TIME_STEP, OUTPUT_SIZE])
        batch_size_ph = tf.placeholder(tf.int32, [])

        #-----input FC layer----#

        #------RNN cells--------#
        if NN_TYPE == 'LSTM':
            y_out = rtb.LSTM_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM, batch_size_ph, activation=None)
        elif NN_TYPE == 'MULTIPLE':
            y_out = rtb.Multi_layer_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM,
                                        CELL_LAYER_NUM, batch_size_ph, activation=None)
        elif NN_TYPE == 'BID':
            y_out = rtb.Bidirect_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM, batch_size_ph, activation=None)

        # if CELL_LAYER_NUM != 1:
        #     stack_rnn_cell = []
        #     for i in range(CELL_LAYER_NUM):
        #         rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=CELL_NUM, state_is_tuple=True)
        #         stack_rnn_cell.append(rnn_cell)
        #     rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stack_rnn_cell, state_is_tuple=True)
        # else:
        #     rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=CELL_NUM, state_is_tuple=True)
        # init_s = rnn_cell.zero_state(batch_size_ph, tf.float32)
        # rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_x, initial_state=init_s)
        #
        # #------output FC layer-----#
        # rnn_ouputs_2D = tf.reshape(rnn_outputs, [-1, 2*CELL_NUM])
        # y_output_2D = tf.layers.dense(rnn_ouputs_2D, OUTPUT_SIZE)
        # y_out = tf.reshape(y_output_2D, [-1, TIME_STEP, OUTPUT_SIZE])

        global_epoch = tf.Variable(0,trainable=False)
        if IS_LR_DECAY:
            learning_rate = tf.train.exponential_decay(LR, global_epoch, TRAIN_EPOCH, 0.96)
        else:
            learning_rate = LR
        loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init_op = tf.global_variables_initializer()

    #---------run session--------#
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        print('Training...')
        for epoch in range(TRAIN_EPOCH):
            global_epoch = epoch
            start = 0
            end = start + TRAIN_BATCH_SIZE
            while(start<=train_x.shape[0]):
                # batch_x, batch_y = get_batch_data(train_x, train_y, TRAIN_BATCH_SIZE)
                # print('Batch_x: ', batch_x)
                # print('Batch_y: ', batch_y)
                # plt.plot(batch_y[:,-1,:])
                # plt.show()

                # feed_dict = {input_x: batch_x, input_y: batch_y, batch_size_ph:TRAIN_BATCH_SIZE}
                # training
                if (end>train_x.shape[0]):                 # the last batch is less than the batch size
                    last_batch_x = train_x[start:train_x.shape[0],:,:]
                    last_batch_y = train_y[start:train_y.shape[0],:,:]
                    feed_dict = {input_x: last_batch_x, input_y: last_batch_y,
                                 batch_size_ph: (train_x.shape[0]-start)}
                    start += TRAIN_BATCH_SIZE
                else:
                    batch_x, batch_y = get_batch_data(train_x, train_y, TRAIN_BATCH_SIZE)
                    feed_dict = {input_x: batch_x, input_y: batch_y,
                                 batch_size_ph: TRAIN_BATCH_SIZE}
                _, lossV, pred = sess.run([train_op, loss, y_out], feed_dict=feed_dict)
            if epoch%50 == 0:
                print('Training epoch %d/%d loss is %f' % (epoch, TRAIN_EPOCH, lossV))
                # print('Learning rate: %f' % lr)

        #     plt.plot(pred[:,-1,:], 'b-')
        #     plt.plot(batch_y[:,-1,:], 'r-')
        #     plt.draw()
        #     plt.pause(0.01)
        # plt.show()
        # print('TRX:', train_x[-1,:,:])
        # print('TRY:', train_y[-1,:,:])
        # print('LBX:', last_batch_x)
        # print('LBY:', last_batch_y)
        # print('TX:',test_x[0,:,:])
        # print('TY:',test_y[0,:,:])
        print('Saving model...')
        saver = tf.train.Saver()
        saver.save(sess,'params', write_meta_graph=False)

else:
    print('Test training...')
    TRAIN_POINTS = train_x.shape[0]
    input_train_x = train_x[0,:,:][np.newaxis,:,:]
    pred_train = np.zeros(shape=(TRAIN_POINTS,1,1))
    lossTT = 0
    with tf.Session() as sess:
        print('Loading model...')
        # reconstruct the model
        input_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
        input_y = tf.placeholder(tf.float32, [None, TIME_STEP, OUTPUT_SIZE])
        batch_size_ph = tf.placeholder(tf.int32, [])

        #-----input FC layer----#

        #------RNN cells--------#
        if NN_TYPE == 'LSTM':
            y_out = rtb.LSTM_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM, batch_size_ph)
        elif NN_TYPE == 'MULTIPLE':
            y_out = rtb.Multi_layer_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM,
                                        CELL_LAYER_NUM, batch_size_ph)
        elif NN_TYPE == 'BID':
            y_out = rtb.Bidirect_RNN(input_x, OUTPUT_SIZE, TIME_STEP, CELL_NUM, batch_size_ph)

        loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
        # loading the model
        saver = tf.train.Saver()
        saver.restore(sess, 'params')

        for i in range(TRAIN_POINTS):
            # input_train_x = train_x[i,:,:][np.newaxis,:,:]
            input_train_y = train_y[i,:,:][np.newaxis,:,:]
            feed_dict = {input_x:input_train_x, input_y:input_train_y, batch_size_ph:1}
            lossV, train_pred = sess.run([loss, y_out], feed_dict=feed_dict)
            lossTT += lossV
            input_train_x = dp.get_test_data(input_train_x, train_pred[:,-1,:])
            pred_train[i,:,:] = train_pred[:,-1,:]

        plt.plot(pred_train[:,0,0],'r-')
        plt.plot(train_y[:,-1,0], 'b-')
        plt.show()
        print('Training loss: %f' % lossTT)

        print('Testing...')
        TEST_POINTS = test_x.shape[0] # depend on the PRED_TIME_RANGE
        input_test_x = test_x[0,:,:][np.newaxis,:,:]

        pred_y = np.zeros(shape=(TEST_POINTS,1,1))
        lossTT = 0
        for i in range(TEST_POINTS):
            # input_test_x = test_x[i,:,:][np.newaxis,:,:]
            input_test_y = test_y[i,:,:][np.newaxis,:,:]
            feed_dict = {input_x:input_test_x, input_y: input_test_y, batch_size_ph:1}
            lossV, test_pred = sess.run([loss, y_out], feed_dict=feed_dict)
            lossTT += lossV
            input_test_x = dp.get_test_data(input_test_x, test_pred[:,-1,:])
            pred_y[i,:,:] = test_pred[:,-1,:]


        plt.plot(pred_y[:,0,0],'r-')
        plt.plot(test_y[:,-1,0], 'b-')
        plt.show()

        print('Testing loss: %f' % lossTT)

    # print('Probability distribution...')
    # prob_mat = np.zeros(shape=(pred_y.shape[0], len(PRED_TIME_RANGE)))
    # for i in range(pred_y.shape[0]):
    #     prob_mat[i,:] = sess.run(calc_prob(pred_y[i,0,0], np.array(PRED_TIME_RANGE)))








