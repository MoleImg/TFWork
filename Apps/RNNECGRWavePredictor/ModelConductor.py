'''
Todo: model graph construction, training and testing
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import RNNToolBox as rtb
import DataParser as dp

def get_batch_data(data_x, data_y, batch_size, start, end):
    # global start, end
    batch_x, batch_y = data_x[start:end,:,:], data_y[start:end,:,:]
    start += batch_size
    end = start + batch_size
    return batch_x, batch_y, start, end

def model_graph(data, labels, batch_size, input_size, output_size, cell_num, time_step, train_epoch,
                nn_type='LSTM', cell_layer_num=None, activation=None, learning_rate=0.01, is_learn_decay=False):
    '''
    use a dictionary to store the whole graph
    :param data: input placeholder
    :param labels: output placeholder
    :param nn_type: model type
    :param batch_size: batch size placeholder
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

    #----input FC layer----#

    #----RNN cells----#
    if nn_type == 'LSTM':
        y_out = rtb.LSTM_RNN(data, output_size, time_step, cell_num, batch_size, activation=None)
    elif nn_type == 'MULTIPLE':
        y_out = rtb.Multi_layer_RNN(data, output_size, time_step, cell_num,
                                    cell_layer_num, batch_size, activation=activation)
    elif nn_type == 'BID':
        y_out = rtb.Bidirect_RNN(data, output_size, time_step, cell_num,
                                batch_size, activation=activation)
    else: # default
        y_out = rtb.LSTM_RNN(data, output_size, time_step, cell_num, batch_size, activation=None)

    global_epoch = tf.Variable(0,trainable=False)
    if is_learn_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_epoch, train_epoch, 0.96)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=y_out)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()
    model_dict['output'] = y_out
    model_dict['loss'] = loss
    model_dict['train_op'] = train_op
    model_dict['init_op'] = init_op

    return model_dict

def model_graph(input_size, output_size, cell_num, time_step,
                nn_type='LSTM', cell_layer_num=None, activation=None, learning_rate=0.01):
    '''
    use a dictionary to store the whole graph
    :param nn_type: model type
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
            y_out = rtb.LSTM_RNN(input_x, output_size, time_step, cell_num, batch_size_ph, activation=None)
        elif nn_type == 'MULTIPLE':
            y_out = rtb.Multi_layer_RNN(input_x, output_size, time_step, cell_num,
                                        cell_layer_num, batch_size_ph, activation=activation)
        elif nn_type == 'BID':
            y_out = rtb.Bidirect_RNN(input_x, output_size, time_step, cell_num,
                                    batch_size_ph, activation=activation)
        else: # default
            y_out = rtb.LSTM_RNN(input_x, output_size, time_step, cell_num, batch_size_ph, activation=None)

        # global_epoch = tf.Variable(0,trainable=False)
        # if is_learn_decay:
        #     learning_rate = tf.train.exponential_decay(learning_rate, global_epoch, train_epoch, 0.96)
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
    model_dict['graph'] = graph

    return model_dict


def train_model(train_x, train_y, model_dict, train_epoch, train_batch_size,learning_rate=0.01,
                is_learn_decay=False, decay_rate=1, is_save=False, save_path=None, checkpoint_path=None):
    '''
    Todo: train the model
    :param train_x:
    :param train_y:
    :param model_dict:
    :param train_epoch:
    :param train_batch_size:
    :param learning_rate:
    :param is_learn_decay:
    :param decay_rate:
    :param is_save: save or not
    :param save_path: path of model be saved
    :return:
    '''

    if is_learn_decay is True: # modify the learning rate in the graph
        graph = model_dict['graph'] # load the graph of the model
        # graph = tf.get_default_graph()
        with graph.as_default():
            global_epoch = tf.Variable(0,trainable=False)
            decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_epoch, train_epoch, decay_rate)
            model_dict['train_op'] = tf.train.AdamOptimizer(decayed_learning_rate).minimize(model_dict['loss']) # modify the training operation
        model_dict['graph'] = graph
    with tf.Session(graph=model_dict['graph']) as sess:
        sess.run(model_dict['init_op'])
        # load the checkpoint of last training
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if checkpoint: # if has previous checkpoints
            saver = tf.train.Saver()
            saver.restore(sess,checkpoint)
            print("[INFO] model restore from the checkpoint {0}".format(checkpoint))


        print('Training...')
        for epoch in range(train_epoch):
            global_epoch = epoch
            # global start, end
            start = 0
            end = start + train_batch_size
            while(start<=train_x.shape[0]):
                # batch_x, batch_y = get_batch_data(train_x, train_y, TRAIN_BATCH_SIZE)
                # print('Batch_x: ', batch_x)
                # print('Batch_y: ', batch_y)
                # plt.plot(batch_y[:,-1,:])
                # plt.show()

                # feed_dict = {input_x: batch_x, input_y: batch_y, batch_size_ph:TRAIN_BATCH_SIZE}
                # training
                if (end>train_x.shape[0]):                 # the last batch is less than the batch size
                    last_batch_x = train_x[start:train_x.shape[0],:,:]
                    last_batch_y = train_y[start:train_y.shape[0],:,:]
                    feed_dict = {model_dict['input_x']: last_batch_x, model_dict['input_y']: last_batch_y,
                                 model_dict['batch_size_ph']: (train_x.shape[0]-start)}
                    start += train_batch_size
                else:
                    batch_x, batch_y, start, end = get_batch_data(train_x, train_y, train_batch_size, start, end)
                    feed_dict = {model_dict['input_x']: batch_x, model_dict['input_y']: batch_y,
                                 model_dict['batch_size_ph']: train_batch_size}
                _, lossV, pred = sess.run([model_dict['train_op'], model_dict['loss'],
                                           model_dict['output']], feed_dict=feed_dict)
            # if (lossV<0.15):
            #     print('[INFO] loss is %f , less than 0.033, BREAK!' % lossV)
            #     break
            if epoch%50 == 0:
                print('Training epoch %d/%d loss is %f' % (epoch, train_epoch, lossV))

        if is_save:
            print('[INFO] Saving model...')
            saver = tf.train.Saver()
            saver.save(sess,save_path, write_meta_graph=False)


def test_model(test_x, test_y, model_dict, checkpoint_path):
    '''
    Todo: test the model
    :param test_x: test data
    :param test_y: test labels
    :param model_dict: model dictionary that store the graph
    :param checkpoint_path: the checkpoint of the model
    :return: predicted results of the input data, with the dimension of (length, 1, 1)
    '''
    TRAIN_POINTS = test_x.shape[0]
    input_test_x = test_x[0,:,:][np.newaxis,:,:]
    test_pred = np.zeros(shape=(TRAIN_POINTS,1,1))
    loss = 0
    with tf.Session(graph=model_dict['graph']) as sess:
        # load the model
        print('[INFO] Loading model...')
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        print("[INFO] model restore from the checkpoint {0}".format(checkpoint))

        print('[INFO] Testing...')
        for i in range(TRAIN_POINTS):
            # input_train_x = train_x[i,:,:][np.newaxis,:,:]
            input_test_y = test_y[i,:,:][np.newaxis,:,:]
            feed_dict = {model_dict['input_x']:input_test_x, model_dict['input_y']:input_test_y,
                         model_dict['batch_size_ph']:1}
            lossV, pred_tmp = sess.run([model_dict['loss'], model_dict['output']], feed_dict=feed_dict)
            loss += lossV
            input_test_x = dp.get_test_data(input_test_x, pred_tmp[:,-1,:])
            test_pred[i,:,:] = pred_tmp[:,-1,:]
        print('[INFO] Testing over...')
        print('Testing loss: %f' % (loss/TRAIN_POINTS))

        return test_pred









