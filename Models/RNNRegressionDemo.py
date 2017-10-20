'''
Using RNN for regression
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_fc_layer(x_in, shape):
    initializer = tf.random_normal_initializer(mean=0, stddev=1.)
    return tf.layers.dense(x_in, shape, kernel_initializer=initializer)

def get_batch(step, time_step):
    global steps
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, time_step)
    batch_x = np.sin(steps)[np.newaxis,:,np.newaxis]
    batch_y = np.cos(steps)[np.newaxis,:,np.newaxis]
    return batch_x, batch_y

# hyper parameters
BATCH_SIZE = 1
TIME_STEP = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_NUM = 32
LR = 0.01

# data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
train_x = np.sin(steps)
train_y = np.cos(steps)

# model
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
    input_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])

    # input layer
    x_in = tf.reshape(input_x, [-1, INPUT_SIZE])
    x_in = add_fc_layer(x_in, CELL_NUM)
    x_in = tf.reshape(x_in, [-1, TIME_STEP, CELL_NUM])

    # RNN
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=32, forget_bias=1.0)
    init_s = rnn_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32) # the first state
    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, input_x, initial_state=init_s, time_major=False)


    # output layer
    output_2D = tf.reshape(rnn_outputs, [-1, CELL_NUM])
    y_out_2D = add_fc_layer(output_2D, OUTPUT_SIZE)
    y_out = tf.reshape(y_out_2D, [-1, TIME_STEP, OUTPUT_SIZE])

    # training
    loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    init_op = tf.global_variables_initializer()

# Traning & testing
with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    print('Training...')
    for step in range(1000):
        batch_x, batch_y = get_batch(step, TIME_STEP)
        if 'final_s' not in globals():   # the first state
            feed_dict={input_x: batch_x, input_y: batch_y}
        else:   # has hidden state
            feed_dict={input_x: batch_x, input_y: batch_y, init_s: final_s}

        _, pred, lossV, final_s = sess.run([train_op, y_out, loss, final_state], feed_dict=feed_dict)

        if step % 50 == 0:
            print('Training step %d/%d.' % (step, 1000))
            print('Loss: %f' % lossV)

        # plotting
    #     plt.plot(steps, batch_y.flatten(), 'r-')
    #     plt.plot(steps, pred.flatten(), 'b-')
    #     plt.ylim((-1.2, 1.2))
    #     plt.draw()
    #     plt.pause(0.05)
    # plt.show()



