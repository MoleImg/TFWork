'''
Todo: model graph construction, training and testing
'''

import tensorflow as tf
import RNNToolBox as rtb

def model_graph(nn_type, input_size, output_size, cell_num, time_step, train_epoch,
                cell_layer_num=None, activation=None, learning_rate=0.01, is_learn_decay=False):
    '''
    use a dictionary to store the whole pragraph
    :param nn_type:
    :param input_size:
    :param output_size:
    :param cell_num:
    :param time_step:
    :param train_epoch:
    :param cell_layer_num:
    :param activation:
    :param learning_rate:
    :param is_learn_decay:
    :return: a model dictionary that store the graph
    '''
    model_dict = {}
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32, [None, time_step, input_size])
        input_y = tf.placeholder(tf.float32, [None, time_step, output_size])
        batch_size_ph = tf.placeholder(tf.int32, [])

        #----input FC layer----#

        #----RNN cells----#
        if nn_type == 'LSTM':
            y_out = rtb.LSTM_RNN(input_x, output_size, time_step, cell_num,
                                 batch_size_ph, activation=activation)
        elif nn_type == 'MULTIPLE':
            y_out = rtb.Multi_layer_RNN(input_x, output_size, time_step, cell_num,
                                        cell_layer_num, batch_size_ph, activation=activation)
        elif nn_type == 'BID':
            y_out = rtb.Bidrect_RNN(input_x, output_size, time_step, cell_num,
                                    batch_size_ph, activation=activation)

    global_epoch = tf.Variable(0,trainable=False)
    if is_learn_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_epoch, train_epoch, 0.96)
    loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_out)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()
    model_dict['input_x'] = input_x
    model_dict['input_y'] = input_y
    model_dict['batch_size_ph'] = batch_size_ph
    model_dict['output'] = y_out
    model_dict['loss'] = loss
    model_dict['train_op'] = train_op
    model_dict['init_op'] = init_op

    return model_dict


# def train_model(data, labels, learning_rate=0.01, is_learn_decay=False):
