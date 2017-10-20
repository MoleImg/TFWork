import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_dataset(data, time_step):
    data_x, data_y = [], []
    for i in range(data.shape[0] - time_step -1):
        x = data[i:(i+time_step),:]
        y = data[(i+1):(i+time_step+1),:]
        data_x.append(x.tolist())
        data_y.append(y.tolist())
    return np.array(data_x), np.array(data_y)

def get_batch(data_x, data_y, batch_size):
    global start, end
    batch_x, batch_y = data_x[start:end,:], data_y[start:end, :]
    return batch_x, batch_y


# ---------------data import------------#
raw_df = pd.read_csv('../Data/httpRequestData.csv')
# plt.plot(raw_df)
# plt.show()
raw_data = raw_df.values
raw_data = raw_data.astype('float32')

# normalization
raw_data_norm = (raw_data - np.mean(raw_data)) / np.std(raw_data)
print('Shape of normalized raw data:', raw_data_norm.shape)
# plt.plot(raw_data_norm)
# plt.show()

# -------------hyper parameters----------#
TIME_STEP = 3
BATCH_SIZE = 5
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_NUM = 64

LR = 0.01
TRAIN_EPOCH = 1500

# -------------Train & Test data---------#
train_data = raw_data_norm[:int(0.8*raw_data_norm.shape[0]),:]
train_x, train_y = create_dataset(train_data, TIME_STEP)
print('Shape of training input:', train_x.shape)
print('Shape of training label:', train_y.shape)
# print(train_x)
test_data = raw_data_norm[int(0.8*raw_data_norm.shape[0]):,:]
test_x, test_y = create_dataset(test_data, TIME_STEP)
print('Shape of testing input:', test_x.shape)
print('Shape of testing label:', test_y.shape)

# -------------Model graph---------------#
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
    input_y = tf.placeholder(tf.float32, [None, TIME_STEP, OUTPUT_SIZE])
    batch_size_ph = tf.placeholder(tf.int32, [])

    # ------input FC layer-------- #

    # ------RNN LSTM cells---------#
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=CELL_NUM)
    init_s = rnn_cell.zero_state(batch_size=batch_size_ph, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_x, initial_state=init_s)

    # ------output FC layer--------#
    rnn_outputs_2D = tf.reshape(rnn_outputs, [-1, CELL_NUM])
    y_out_2D = tf.layers.dense(rnn_outputs_2D, OUTPUT_SIZE)
    y_out = tf.reshape(y_out_2D, [-1, TIME_STEP, OUTPUT_SIZE])

    # ------training configurations------#
    loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    init_op = tf.global_variables_initializer()

# --------------Run session--------------#
with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    print('Training...')
    for epoch in range(TRAIN_EPOCH):
        start = 0
        end = start+BATCH_SIZE
        batch_num = 0
        while(end<train_x.shape[0]):
            # print('Batch NO.', batch_num+1)
            batch_x, batch_y = get_batch(train_x, train_y, BATCH_SIZE)
            # print(batch_num,':',batch_x)
            # print(batch_num,':',batch_y)
            start += BATCH_SIZE
            end = start + BATCH_SIZE
            # training
            if 'final_s' not in globals():                 # first state, no any hidden state
                feed_dict = {input_x: batch_x, input_y: batch_y, batch_size_ph: BATCH_SIZE}
            else:                                           # has hidden state, so pass it to rnn
                feed_dict = {input_x: batch_x, input_y: batch_y, batch_size_ph: BATCH_SIZE, init_s: final_s}

            _, lossV, pred, final_s = sess.run([train_op, loss, y_out, final_state], feed_dict=feed_dict)
            batch_num += 1
        if epoch % 50 == 0:
            print('Training epoch %d/%d: loss is %f' % (epoch, TRAIN_EPOCH, lossV))
    #     plt.plot(pred[:,-1,:], 'b-')
    #     plt.plot(batch_y[:,-1,:], 'r-')
    #     plt.draw()
    #     plt.pause(0.01)
    # plt.show()

    print('Testing...')
    test_pred = np.arange(test_x.shape[0]).reshape((test_x.shape[0],1))
    for i in range(test_x.shape[0]):
        input_test = test_x[i,:,:][np.newaxis,:,:]
        output_test = test_y[i,:,:][np.newaxis,:,:]
        lossTV, predT = sess.run([loss, y_out], feed_dict={
            input_x: input_test, input_y: output_test, batch_size_ph: 1})
        print('Testing time %d, loss is %f' % (i+1, lossTV))
        test_pred[i,:] = predT[:,-1,:]


    plt.plot(test_pred, 'b-')
    plt.plot(test_y[:,-1,:], 'r-')
    plt.show()

