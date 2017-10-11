'''
Demo for the preparation for the ECG-RNN exp.
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# hyper parameters
BATCH_SIZE = 64
TIME_STEP = 28 # rnn time step for all
INPUT_SIZE = 28 # rnn input size for each step
CLASS_NUM = 10
LR = 0.01
EPOCH = 1200

# data import
MNIST_PATH = '../MNIST_data'
raw_data = input_data.read_data_sets(MNIST_PATH, one_hot=True) # training data
# testing data
test_x = raw_data.test.images[:2000]
test_y = raw_data.test.labels[:2000]

# model
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, shape=[None, TIME_STEP*INPUT_SIZE])
    input_image = tf.reshape(input_x, [-1, TIME_STEP, INPUT_SIZE])
    input_y = tf.placeholder(tf.int32, shape=[None, CLASS_NUM])

    # rnn
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, input_image,
                                        initial_state=None, dtype=tf.float32, time_major=False)
    pred_y = tf.layers.dense(outputs[:,-1,:], 10)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=pred_y)
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

    acc = tf.metrics.accuracy( # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(input_y, axis=1), predictions=tf.argmax(pred_y, axis=1))[1]

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    print('Training...')
    for epoch in range(EPOCH):
        batch_train_x, batch_train_y = raw_data.train.next_batch(BATCH_SIZE)
        _, lossV, accV = sess.run([optimizer, loss, acc], feed_dict={input_x: batch_train_x, input_y: batch_train_y})

        if epoch % 50 == 0:
            print('Training batch %d/%d' % (epoch, EPOCH))
            print('Loss: %f and accuracy: %f' % (lossV, accV))

    print('Testing...')
    accTest = sess.run(acc, feed_dict={input_x: test_x, input_y: test_y})
    print('Testing accuracy: %f' % (accTest))

