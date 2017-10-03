import numpy as np
import tensorflow as tf


# raw data
x_data = np.arange(20).reshape((1,20))
print(x_data)
y_data = 3.5*x_data+2.5
print(y_data)

train_x = x_data[:, 0:15]
train_y = y_data[:, 0:15]
test_x = x_data[:, 15:]
test_y = y_data[:, 15:]

graph = tf.Graph()
with graph.as_default():
    x_input = tf.placeholder(dtype=tf.float32, shape=(1,None))
    y_input = tf.placeholder(dtype=tf.float32, shape=(1,None))

    weights = tf.Variable(1.0)
    biases = tf.Variable(0.1)

    y_pred = weights * x_input + biases
    loss = tf.reduce_mean(tf.square(y_pred - y_input))
    optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)


iter_num = 100
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('Training...')
    for iter in range(iter_num):
        feed_dict = {x_input: train_x, y_input: train_y}
        _, lossV, weightsV, biasesV = sess.run([optimizer, loss, weights, biases], feed_dict=feed_dict)
        print('Iter: ', iter)
        print('Loss: ', lossV)
        print('Weights: ', weightsV)
        print('Biases: ', biasesV)

    print('Testing...')
    feed_dict = {x_input: test_x, y_input: test_y}
    _, y_test = sess.run([optimizer, y_pred], feed_dict=feed_dict)
    print('test_y: ', test_y)
    print('y_test: ', y_test)