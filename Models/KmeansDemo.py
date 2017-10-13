import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

def one_hot_encoding(input, cluster):
    output = np.zeros([1, cluster], dtype=np.int32)
    output[:,input[0]] = 1
    return output



# import data
from sklearn import datasets

raw_data = datasets.load_iris()
data_x = raw_data.data
data_y = raw_data.target
#
# print(data_x.shape)
# print('==================')
# print(data_y.shape)

# parameters
class_num = 3
feature_num = 4
train_steps = 50

# one-hot encoding
data_y = np.reshape(data_y, (data_y.shape[0], 1))
data_y_onehot = np.ndarray((data_y.shape[0], 3), dtype=np.int32)
# print(data_y_onehot.shape)
for i in range(data_y_onehot.shape[0]):
    data_y_onehot[i,:] = one_hot_encoding(data_y[i,:], 3)

# print(data_y_onehot.shape)

# split data
from sklearn.cross_validation import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)

# Graph
graph = tf.Graph()
with graph.as_default():
    x_input = tf.placeholder(tf.float32, shape=[None, feature_num])
    y_input = tf.placeholder(tf.int32, shape=[None, class_num])

    # Model
    model = KMeans(inputs=x_input, num_clusters=class_num)
