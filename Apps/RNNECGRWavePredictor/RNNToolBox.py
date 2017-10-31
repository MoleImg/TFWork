'''
TODO: Integrtion of different types of RNN
include:
Single layer LSTM
Multi-layer LSTM
Binary directional RNN
'''
import numpy as np
import tensorflow as tf


def LSTM_RNN(data, output_size, time_step, cell_num, batch_size, activation=None):
    '''
    TODO: LSTM RNN
    :param data: input x data
    :param output_size:
    :param time_step:
    :param cell_num: number of the cells
    :param batch_size: batch size of each iteration of training
    :param activation: if None, use linear activation function
    :return: LSTM RNN output
    '''
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_num, state_is_tuple=True)
    init_s = rnn_cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell,data,initial_state=init_s)

    #-----output FC layer----#
    rnn_outputs_2D = tf.reshape(rnn_outputs, [-1, cell_num])
    y_output_2D = tf.layers.dense(rnn_outputs_2D, output_size, activation=activation)
    y_out = tf.reshape(y_output_2D, [-1, time_step, output_size])
    return y_out


def Multi_layer_RNN(data, output_size, time_step, cell_num,
                    layer_num, batch_size, activation=None):
    '''
    Multi-layer RNN
    :param data: input x data
    :param output_size:
    :param time_step:
    :param layer_num: layer numbers of the multi-layers
    :param cell_num: number of the cells
    :param batch_size:
    :param activation: if None, use linear activation funcion
    :return: Multi-layer RNN output
    '''
    if layer_num  == 1:
        return LSTM_RNN(data, output_size, time_step, cell_num,
                    batch_size, activation=None)
    else:
        stack_rnn_cell = []
        for i in range(layer_num):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_num, state_is_tuple=True)
            stack_rnn_cell.append(rnn_cell)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stack_rnn_cell, state_is_tuple=True)
        init_s = rnn_cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, data, initial_state=init_s)
        #----output FC layer----#
        rnn_outputs_2D = tf.reshape(rnn_outputs, [-1, cell_num])
        y_output_2D = tf.layers.dense(rnn_outputs_2D, output_size, activation=activation)
        y_out = tf.reshape(y_output_2D, [-1, time_step, output_size])
        return y_out


def Bidirect_RNN(data, output_size, time_step, cell_num,
                    batch_size, activation=None):
    '''
    Binary directional RNN
    :param data:
    :param output_size:
    :param time_step:
    :param cell_num:
    :param batch_size:
    :param activation:
    :return: Binary-layer RNN output
    '''
    rnn_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_num)
    rnn_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_num)
    init_s_fw = rnn_cell_fw.zero_state(batch_size, tf.float32)
    init_s_bw = rnn_cell_bw.zero_state(batch_size, tf.float32)
    rnn_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, data,
                                                             initial_state_fw=init_s_fw,initial_state_bw=init_s_bw)
    rnn_outputs = tf.concat(rnn_outputs, 2)
    rnn_outputs = tf.transpose(rnn_outputs, [1,0,2])
    #----output FC layer----#
    rnn_outputs_2D = tf.reshape(rnn_outputs, [-1, 2*cell_num])
    y_output_2D = tf.layers.dense(rnn_outputs_2D, output_size, activation=activation)
    y_out = tf.reshape(y_output_2D, [-1, time_step, output_size])
    return y_out


