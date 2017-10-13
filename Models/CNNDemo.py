import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def calc_accuracy(labels, preds):
    return 100*(np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / labels.shape[0])

# hyper-parameters
LR = 0.01
BATCH_SIZE = 50
HIDDEN = 128

feature_num = 28*28
class_num = 10
epoch_num = 500

# data import
mnist_path = '../MNIST_data'
mnist = input_data.read_data_sets(mnist_path, one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# Model
graph = tf.Graph()
with graph.as_default():
    x_input = tf.placeholder(tf.float32, shape=[None, feature_num]) / 255
    image = tf.reshape(x_input, [-1, 28, 28, 1])
    y_input = tf.placeholder(tf.float32, shape=[None, class_num])
    # conv+pooling layer
    conv1 = tf.layers.conv2d(image, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
    # conv+pooling layer
    conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
    # fc layer
    fc_input = tf.contrib.layers.flatten(pool2)
    hidden1 = tf.layers.dense(fc_input, HIDDEN, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden1, class_num)
    # output
    output = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


# run session
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print('Training...')
    for epoch in range(epoch_num):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        feed_dict = {x_input: batch_x, y_input: batch_y}
        _, lossV, outputV = sess.run([optimizer, loss, output], feed_dict=feed_dict)

        if epoch%50 == 0:
            print('Epoch %d/%d' % (epoch, epoch_num))
            print('Loss: %.4f' % (lossV))
            print('Accuracy: %.4f' % (calc_accuracy(batch_y, outputV)))

    print('Testing...')
    feed_dict={x_input: test_x, y_input: test_y}
    pred_y = sess.run(output, feed_dict=feed_dict)
    print('Testing accuracy: %.4f' % (calc_accuracy(test_y, pred_y)))






