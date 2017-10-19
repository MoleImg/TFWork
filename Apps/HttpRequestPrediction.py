import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_dataset(data, time_step):
    data_x, data_y = [], []
    for i in range(len(data)-time_step-1):
        x = data[i:(i+time_step), 0]
        y = data[(i+1):(i+time_step+1), 0]
        data_x.append(x)
        data_y.append(y)

    return np.array(data_x), np.array(data_y)



# raw data
raw_df = pd.read_csv('../Data/httpRequestData.csv')
# plt.plot(raw_df)
# plt.show()

raw_data = raw_df.values
raw_data = raw_data.astype('float32')

# Normalization
# scaler = MinMaxScaler(feature_range=(0, 1))
# raw_data = scaler.fit_transform(raw_data)
print(raw_data.shape)

# hyper parameters
TRAIN_SIZE = int(len(raw_data) * 0.8)
TEST_SIZE = len(raw_data) - TRAIN_SIZE
TIME_STEP = 3
INPUT_SIZE = 1
OUTPUT_SIZE = 1
BATCH_NUM = 75
BATCH_SIZE = 5
CELL_NUM = 64
LR = 0.01

TRAINING_EPOCH = 100


# train & test data
train_data = raw_data[0:TRAIN_SIZE, :]
test_data = raw_data[TRAIN_SIZE:len(raw_data), :]
train_x, train_y = create_dataset(train_data, TIME_STEP)
test_x, test_y = create_dataset(test_data, TIME_STEP)
train_x = np.reshape(train_x, (train_x.shape[0], TIME_STEP, INPUT_SIZE))
train_y = np.reshape(train_y, (train_y.shape[0], TIME_STEP, OUTPUT_SIZE))
test_x = np.reshape(test_x, (test_x.shape[0], TIME_STEP, INPUT_SIZE))
test_y = np.reshape(test_y, (test_y.shape[0], TIME_STEP, INPUT_SIZE))
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

# model
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
    input_y = tf.placeholder(tf.float32, [None, TIME_STEP, OUTPUT_SIZE])
    # input fc layer
    x_in = tf.reshape(input_x, [-1, INPUT_SIZE])
    x_in = tf.layers.dense(x_in, CELL_NUM)
    # RNN cells
    x_in = tf.reshape(x_in, [-1, TIME_STEP, CELL_NUM])
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=CELL_NUM, forget_bias=1.0)
    init_s = rnn_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_x, initial_state=init_s)
    # y_out = rnn_outputs[:,:,-1][:,:,np.newaxis]
    # output fc layer
    output_2D = tf.reshape(rnn_outputs, [-1, CELL_NUM])
    y_out_2D = tf.layers.dense(output_2D, OUTPUT_SIZE)
    y_out = tf.reshape(y_out_2D, [-1, TIME_STEP, OUTPUT_SIZE])


    # training setups
    loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    init_op = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    print('Training...')
    for epoch in range(TRAINING_EPOCH):
        test_pred = np.arange(BATCH_NUM).reshape((BATCH_NUM,1))
        test_true = np.arange(BATCH_NUM).reshape((BATCH_NUM,1))
        start = 0
        end = start + BATCH_SIZE
        while(end<BATCH_NUM):
            steps = np.linspace(start, end, 15)
            batch_x, batch_y = train_x[start:end, :, :], train_y[start:end, :, :]
            # batch_x, batch_y = batch_x[np.newaxis, :, :], batch_y[np.newaxis, :, :]
            print('Batch_x: ', batch_x)
            print('Batch_y: ', batch_y)
            start += BATCH_SIZE
            end += BATCH_SIZE
            if 'final_s' not in globals():
                feed_dict={input_x: batch_x, input_y: batch_y}
            else:
                feed_dict={input_x: batch_x, input_y: batch_y, init_s:final_s}
            _, lossV, pred, final_s = sess.run([train_op, loss, y_out, final_state], feed_dict=feed_dict)

            # test_pred[i,:] = pred[:, -1, :]
            # test_true[i,:] = batch_y[:, -1, :]
            plt.plot(steps, batch_y.flatten(), 'r-')
            plt.plot(steps, pred.flatten(), 'b-')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.1)
        plt.show()


        if (epoch+1)%10 == 0:
            print('Training epoch %d/%d. Loss: %f' % (epoch+1, TRAINING_EPOCH, lossV))
            plt.plot(test_pred)
            plt.plot(test_true)
            plt.show()

    # print('Testing...')
    # test_pred = np.empty_like(test_y)
    # for i in range(test_x.shape[0]):
    #     feed_dict={input_x: test_x[i,:,:][np.newaxis,:,:], input_y: test_y[i,:][np.newaxis,:], init_s: final_s}
    #     lossTV, predT = sess.run([loss, y_out], feed_dict=feed_dict)
    #     # predT.reshape((1,))
    #     test_pred[i,:] = predT
    #     print('Testing loss: %f' % lossTV)
    #     print('Testing result: %f' % predT)
    #     print('Truth: %f' % test_y[i,:])

# test_pred = np.array(test_pred)
# np.reshape(test_pred, (test_pred.shape[0],1))
# test_pred = scaler.inverse_transform(test_pred)
# test_y = scaler.inverse_transform(test_y)

# plt.plot(test_y)
# plt.plot(test_pred)
# plt.show()




