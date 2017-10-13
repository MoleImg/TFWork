# Autoencoder

'''
Autoencoder construction
Copyright: Dongqiudi
'''

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

'''
Function definition
'''

def addCoderLayer(inputs, weights, biases, AF=None):
    '''
    Single Encoder/Decoder Layer
    '''
    tmp = tf.add(tf.matmul(inputs, weights), biases)
    if AF is None:
        return tmp
    else:
        return AF(tmp)


'''
Parameter configuration
'''
# Input data
mnistRawData = input_data.read_data_sets('MNIST_data', one_hot=True)
imageSize = 28
# Training batch
hm_epoches = 1
batchSize = 100
trainingBatchNum = int(mnistRawData.train.num_examples / batchSize)
# Testing and displaying
examplesToShow = 10
# Network Sturcture
nInput = imageSize * imageSize
hidLayer1NodeNum = 256
hidLayer2NodeNum = 128
# Input placeholder
xInputHandle = tf.placeholder(tf.float32, [None, nInput])
'''
Graph configuraion
'''
# Weights & Biases
Weights = {
    'encoderH1': tf.Variable(tf.random_normal([nInput, hidLayer1NodeNum])),
    'encoderH2': tf.Variable(tf.random_normal([hidLayer1NodeNum, hidLayer2NodeNum])),
    'decoderH1': tf.Variable(tf.random_normal([hidLayer2NodeNum, hidLayer1NodeNum])),
    'decoderH2': tf.Variable(tf.random_normal([hidLayer1NodeNum, nInput])),
}

Biases = {
    'encoderH1': tf.Variable(tf.random_normal([hidLayer1NodeNum])),
    'encoderH2': tf.Variable(tf.random_normal([hidLayer2NodeNum])),
    'decoderH1': tf.Variable(tf.random_normal([hidLayer1NodeNum])),
    'decoderH2': tf.Variable(tf.random_normal([nInput])),
}


# Two-layer network
with tf.name_scope('encoderLayer1'):
    outputEncoder1 = addCoderLayer(xInputHandle, Weights['encoderH1'], Biases['encoderH1'], AF=tf.nn.sigmoid)
with tf.name_scope('encoderLayer2'):
    outputEncoder2 = addCoderLayer(outputEncoder1, Weights['encoderH2'], Biases['encoderH2'], AF=tf.nn.sigmoid)
with tf.name_scope('decoderLayer1'):
    outputDecoder1 = addCoderLayer(outputEncoder2, Weights['decoderH1'], Biases['decoderH1'], AF=tf.nn.sigmoid)
with tf.name_scope('decoderLayer2'):
    outputFinal = addCoderLayer(outputDecoder1, Weights['decoderH2'], Biases['decoderH2'], AF=tf.nn.sigmoid)

# Loss & Training
with tf.name_scope('Loss'):
    Loss = tf.reduce_mean(tf.square(xInputHandle - outputFinal), reduction_indices=[1])
with tf.name_scope('Training'):
    TrainStep = tf.train.AdamOptimizer().minimize(Loss)

# Initialization
init = tf.initialize_all_variables()

'''
Run session
'''
with tf.Session() as sess:
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        train_writer = tf.train.SummaryWriter('log/train', sess.graph)
        test_writer = tf.train.SummaryWriter('log/test', sess.graph)
    else: # tensorflow version >= 0.12
        train_writer = tf.summary.FileWriter("log/train", sess.graph)
        test_writer = tf.summary.FileWriter("log/test", sess.graph)
    sess.run(init)
    # training
    for epoch in range(hm_epoches):
        for step in range(trainingBatchNum):
            batchXData, batchYData = mnistRawData.train.next_batch(batchSize)
            cost, _ = sess.run([Loss, TrainStep], feed_dict={xInputHandle: batchXData})

        if epoch % 1 == 0:
            print('Epoch', epoch+1, 'training completed with the loss of', cost[-1])
            # Testing
            cost,_ = sess.run([Loss, outputFinal], feed_dict={xInputHandle: mnistRawData.test.images[:examplesToShow]})
            print('Epoch', epoch+1, 'testing completed with the loss of', cost[-1])
    
    print('Finished!')

