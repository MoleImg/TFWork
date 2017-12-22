'''
Todo: Implementation demo of random forest
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data

# import data
mnistRawData = input_data.read_data_sets('MNIST_data', one_hot=False)

# parameters
num_steps = 500
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, shape=[None, num_features])
    input_y = tf.placeholder(tf.int32, shape=[None])

    # random forest parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features,
                                          num_trees=num_trees, max_nodes=max_nodes).fill()

    # bulid the random forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)

    train_op = forest_graph.training_graph(input_x, input_y)
    loss = forest_graph.training_loss(input_x, input_y)

    # measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(input_x)
    corr_pred = tf.equal(tf.argmax(infer_op, 1), tf.cast(input_y, tf.int64))
    acc_op = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

    # initialization of variables and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
                         resources.initialize_resources(resources.shared_resources()))
with tf.train.MonitoredSession() as sess:
    # initilization
    sess.run(init_vars)
    # training
    for batch in range(1, num_steps+1):
        batch_x, batch_y = mnistRawData.next_batch(batch_size)
        _, lossV = sess.run([train_op, loss], feed_dict={input_x: batch_x, input_y: batch_y})
        if batch % 50 == 0:
            acc = sess.run(acc_op, feed_dict={input_x: batch_x, input_y: batch_y})
            print('Step: %d, Loss: %f, Acc: %f' % (batch, lossV, acc))

    # test
    test_x, test_y = mnistRawData.test.images, mnistRawData.test.labels
    print('Test accuracy:', sess.run(acc_op, feed_dict={input_x: test_x, input_y: test_y}))