import numpy as np
import tensorflow as tf

# Hyper parameters
N_SAMPLES = 20
N_HIDDEN = 500

# data import
train_x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
train_y = train_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# test data
test_x = train_x.copy()
test_y = test_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# Model
graph = tf.Graph()
with graph.as_default():
    x_input = tf.placeholder(tf.float32, [None, 1])
    y_input = tf.placeholder(tf.float32, [None, 1])
    training_flag = tf.placeholder(tf.bool, None)

    # overfitting net
    o1 = tf.layers.dense(x_input, N_HIDDEN, tf.nn.relu)
    o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
    o_out = tf.layers.dense(o2, 1)
    o_loss = tf.losses.mean_squared_error(y_input, o_out)
    o_optimizer = tf.train.AdamOptimizer(0.01).minimize(o_loss)

    # dropout net
    d1 = tf.layers.dense(x_input, N_HIDDEN, tf.nn.relu)
    d1 = tf.layers.dropout(d1, rate=0.5, training=training_flag)
    d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
    d2 = tf.layers.dropout(d2, rate=0.5, training=training_flag)
    d_out = tf.layers.dense(d2, 1)
    d_loss = tf.losses.mean_squared_error(y_input, d_out)
    d_optimizer = tf.train.AdamOptimizer(0.01).minimize(d_loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    # training
    for i in range(500):
        feed_dict={x_input: train_x, y_input: train_y, training_flag: True}
        _, _, o_lossV, d_lossV = sess.run([o_optimizer, d_optimizer, o_loss, d_loss], feed_dict=feed_dict)

        if i%50 == 0:
            print('Step %d/500:' % (i))
            print('o_loss: %f', (o_lossV))
            print('d_loss: %f', (d_lossV))

    # testing
    feed_dict={x_input: test_x, y_input:test_y, training_flag:False}
    o_lossV, d_lossV = sess.run([o_loss, d_loss], feed_dict=feed_dict)