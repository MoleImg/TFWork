import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def calc_prob(predict_data, time_range):
    '''
    calculate the probability distribution of the specific time range
    :param predict_data: the data predicted by RNN model
    :param time_range: the time window range of the probability (numpy array)
    :return: independent probability distribution on the time_range, corresponding to the predicted data
    '''
    prob = (1-tf.nn.softmax(abs(time_range-predict_data))) / time_range.shape[0]
    return prob

with tf.Session() as sess:
    pred_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    time_range = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    yyy = np.zeros((len(pred_data), len(time_range)))
    for i in range(len(pred_data)):
        yyy[i,:] = sess.run(calc_prob(pred_data[i], np.array(time_range)))
    print(yyy)

for i in range(len(pred_data)):
    plt.plot(yyy[i,:])
plt.show()