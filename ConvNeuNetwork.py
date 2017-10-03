# Convolutional Neural Network

'''
Convolutional neural network construction
Copyright: Dongqiudi
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
'''
Tensorflow configuration
'''

'''
Function definition
'''
def calcCrossEntropy1D(DataA, DataB):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(DataA, DataB))

def calcMeanSqureError1D(DataA, DataB):
    return tf.reduce_mean(reduce_sum(tf.square(DataA - DataB), reduction_indices=[1]))

def weightsVarDef(shape):
    '''
    Initialization of weights
    '''
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.1))

def biasesVarDef(shape):
    '''
    Initialization of biases
    '''
    return tf.Variable(tf.constant(0.1, shape=shape))

def addConvLayer2d(xInput, weights, biases, strides, padding='SAME', AF=None):
    '''
    Adding convolutional layer
    '''
    convOut = tf.nn.conv2d(xInput, weights, strides, padding)
    if AF is None:
        return convOut + biases
    else:
        return AF(convOut + biases)

def addPoolLayer2d(xInput, kSize, strides, padding='SAME'):
    '''
    Adding pooling layer
    '''
    return tf.nn.max_pool(xInput, kSize, strides, padding)

def addFuncLayer(xInput, weights, biases, AF=None):
    '''
    Adding functional layer
    '''
    with tf.name_scope('WxPlusB'):
        WxPlusB = tf.matmul(xInput, weights) + biases
    if AF is None:
        return WxPlusB
    else:
        return AF(WxPlusB)

def calcAccuracySess(groundTruthData, groundTruthLabel, keepProbStack):
    global outputFinal
    prediction = sess.run(outputFinal, feed_dict={xInputHandle:groundTruthData, keepProb:keepProbStack})
    correctPre = tf.equal(tf.argmax(prediction,1), tf.argmax(groundTruthLabel,1))
    accuracy = tf.reduce_mean(tf.cast(correctPre, tf.float32))
    result = sess.run(accuracy, feed_dict={xInputHandle: groundTruthData, yInputHandle: groundTruthLabel, keepProb: keepProbStack})
    return result

def calcAccuracy(prediction, truthLabel):
    correctPre = tf.equal(tf.argmax(prediction,1), tf.argmax(truthLabel,1))
    return tf.reduce_mean(tf.cast(correctPre, tf.float32))

'''
Parameter configuration
'''
inputImageSize = 28
inputImageDepth = 1
inputSize1D = inputImageSize*inputImageSize
outputSize1D = 10

convLayerNum = 2
convMoveStride = 1 # size of conv window movement
convPoolStride = 2 # size of pooling window movement
convPoolKSize = 2 # size of pooling window
patchSize = 5
outPutSizeConvLayer1 = 32
outputSizeConvLayer2 = 64

funcLayer1NodeNum = int(inputImageSize/4)*int(inputImageSize/4)*outputSizeConvLayer2
funcLayer2NodeNum = 1024


'''
Input placeholders
'''
with tf.name_scope('Inputs'):
    xInputHandle = tf.placeholder(tf.float32,shape=[None, inputImageSize*inputImageSize])/255. 
    yInputHandle = tf.placeholder(tf.float32, shape=[None, outputSize1D])
with tf.name_scope('InputsReshape'):
    imInput = tf.reshape(xInputHandle, [-1, inputImageSize, inputImageSize, 1])

keepProb = tf.placeholder(tf.float32)

'''
Network construction
'''
# 1st convolutional layer
with tf.name_scope('ConvolutionLayer1'):
    with tf.name_scope('Weights'):
        weights1 = weightsVarDef([patchSize, patchSize, inputImageDepth, outPutSizeConvLayer1])
    with tf.name_scope('Biases'):
        biases1 = biasesVarDef([outPutSizeConvLayer1])
    with tf.name_scope('Convolution'):
        convOutput1 = addConvLayer2d(imInput, weights1, biases1, [1, convMoveStride, convMoveStride, 1], AF=tf.nn.relu)

# 1st pooling layer
with tf.name_scope('PoolingLayer1'):
    poolOutput1 = addPoolLayer2d(convOutput1, [1, convPoolKSize, convPoolKSize, 1], 
                        [1, convPoolStride, convPoolStride, 1])

# 2nd convolutional layer
with tf.name_scope('ConvolutionLayer2'):
    with tf.name_scope('Weights'):
        weights2 = weightsVarDef([patchSize, patchSize, outPutSizeConvLayer1, outputSizeConvLayer2])
    with tf.name_scope('Biases'):
        biases2 = biasesVarDef([outputSizeConvLayer2])
    with tf.name_scope('Convolution'):
        convOutput2 = addConvLayer2d(poolOutput1, weights2, biases2, [1, convMoveStride, convMoveStride, 1], AF=tf.nn.relu)

# 2nd pooling layer
with tf.name_scope('PoolingLayer2'):
    poolOutput2 = addPoolLayer2d(convOutput2, [1, convPoolKSize, convPoolKSize, 1], 
                [1, convPoolStride, convPoolStride, 1])

# 1st functional layer
# Reshape the image into one dimension
with tf.name_scope('Reshape'):
    funcInput1 = tf.reshape(poolOutput2, [-1, funcLayer1NodeNum])

with tf.name_scope('FunctionalLayer1'):
    with tf.name_scope('Weights'):
        weights3 = weightsVarDef([funcLayer1NodeNum, funcLayer2NodeNum])
    with tf.name_scope('Biases'):
        biases3 = biasesVarDef([funcLayer2NodeNum])
    funcOutput1 = addFuncLayer(funcInput1, weights3, biases3, AF=tf.nn.relu)
    funcOutput1 = tf.nn.dropout(funcOutput1, keepProb)

# 2nd functional layer
with tf.name_scope('FunctionalLayer2'):
    with tf.name_scope('Weights'):
        weights4 = weightsVarDef([funcLayer2NodeNum, outputSize1D])
    with tf.name_scope('Biases'):
        biases4 = biasesVarDef([outputSize1D])
    outputFinal = addFuncLayer(funcOutput1, weights4, biases4, AF=tf.nn.softmax)

# Input data
mnistRawData = input_data.read_data_sets('MNIST_data', one_hot=True)
# Training
# Training parameters
# learningRate = tf.constant(0.01)
batchSize = 100
trainingBatchNum = int(mnistRawData.train.num_examples / batchSize)
hm_epoches = 1
keepProbTrain = 1
keepProbCompare = 1
with tf.name_scope('Loss'):
    lossTensor = calcCrossEntropy1D(outputFinal, yInputHandle)
    tf.scalar_summary('Loss', lossTensor)
    accuracy = calcAccuracy(outputFinal, yInputHandle)
    tf.scalar_summary('Accuracy', accuracy)
with tf.name_scope('Training'):
    trainStep = tf.train.AdamOptimizer().minimize(lossTensor)

# Initialization
init = tf.initialize_all_variables()
# Merging
merge = tf.merge_all_summaries()
'''
Session running
'''
with tf.Session() as sess:
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        train_writer = tf.train.SummaryWriter('log/train', sess.graph)
        test_writer = tf.train.SummaryWriter('log/test', sess.graph)
    else: # tensorflow version >= 0.12
        train_writer = tf.summary.FileWriter("log/train", sess.graph)
        test_writer = tf.summary.FileWriter("log/test", sess.graph)
    # initialization
    sess.run(init)

    # training
    for epoch in range(hm_epoches):
        for step in range(trainingBatchNum):
            # training data
            batchXData, batchYData = mnistRawData.train.next_batch(batchSize)
            sess.run(trainStep, feed_dict={xInputHandle: batchXData, yInputHandle: batchYData, keepProb:keepProbTrain})
            if step == trainingBatchNum-1:
                print('Epoch', epoch+1, 'training completed with the training accuracy of ', 
                        calcAccuracySess(batchXData, batchYData, keepProbCompare))
                print('Epoch', epoch+1, 'training completed with the testing accuracy of ',
                        calcAccuracySess(mnistRawData.test.images, mnistRawData.test.labels, keepProbCompare))
'''               
                train_rs = sess.run(merge, 
                        feed_dict={xInputHandle: batchXData, yInputHandle: batchYData, keepProb:keepProbCompare})
                train_writer.add_summary(train_rs, epoch)
                test_rs = sess.run(merge, 
                        feed_dict={xInputHandle: mnistRawData.test.images, yInputHandle: mnistRawData.test.labels, keepProb:keepProbCompare})
                test_writer.add_summary(test_rs, epoch)
'''