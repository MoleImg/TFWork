import numpy as np
import tensorflow as tf

def one_hot_encoding(input, cluster):
    output = np.zeros([1, cluster], dtype=np.float32)
    output[:, input[0]] = 1
    return output

def one_hot_decoding(input, cluster):
    output = input.nonzero()
    return output[0]

def calc_accuracy(pred, ref):
    return (100*np.sum(np.argmax(pred, 1) == np.argmax(ref, 1))/(pred.shape[0]))

# data import
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

raw_data = load_iris()
data_x = raw_data.data
data_y = raw_data.target

# parameters
class_num = 3
feature_num = 4
epoch_num = 100

# one-hot encoding
data_y = data_y.reshape((data_y.shape[0], 1))
data_y_onehot = np.zeros([data_x.shape[0], class_num])
for i in range(data_x.shape[0]):
    data_y_onehot[i,:] = one_hot_encoding(data_y[i], class_num)

# print(data_y_onehot)
# print(data_y_onehot.shape)

# data split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_onehot, test_size=0.25)

# Graph
graph = tf.Graph()
with graph.as_default():
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, feature_num])
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, class_num])

    weights = tf.Variable(tf.truncated_normal(shape=[feature_num, class_num]))
    biases = tf.Variable(tf.truncated_normal(shape=[class_num]))

    logits = tf.add(tf.matmul(x_input, weights), biases)
    y_pred = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    sess.run(init)
    print('Training...')
    for i in range(epoch_num):
        feed_dict = {x_input: train_x, y_input: train_y}
        _, lossV, y_predT = sess.run([optimizer, loss, y_pred], feed_dict=feed_dict)

        print('Epoch %d/%d with the loss of %f' % (i, epoch_num, lossV))
        print('Training accuracy: %f' % calc_accuracy(y_predT, train_y))

    print('Testing...')
    feed_dict = {x_input: test_x, y_input: test_y}
    y_predV = sess.run(y_pred, feed_dict=feed_dict)

    print('Testing accuracy: %f' % calc_accuracy(y_predV, test_y))

