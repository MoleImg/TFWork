# Fundamental Deep Neural Network

'''
Fundamental deep neural network construction
Standard procedure
Copyright: Dongqiudi
'''
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
'''
Tensorflow configuration
'''
def calcCrossEntropy1D(DataA, DataB):
    '''
    TODO: calculate the cross entropy under the tensorflow architecture
    '''
    return  tf.reduce_mean(-tf.reduce_sum(DataB * tf.log(DataA), reduction_indices=[1]))

def calcMeanSqureError1D(DataA, DataB):
    '''
    TODO: calculate the mean square error under the tensorflow architecture
    '''
    return tf.reduce_mean(reduce_sum(tf.square(DataA - DataB), reduction_indices=[1]))

def addLayer(layerName, inputs, inputSize, outputSize, keepProb, actFunc=None):
    '''
    TODO: construct a layer of the neural network
    '''
    with tf.name_scope(layerName):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([inputSize, outputSize]), name='Weighting')
            # tf.histogram_summary(layerName + '/Weights', Weights)
        with tf.name_scope('Biases'):
            Biases = tf.Variable(tf.random_normal([1, outputSize]) + 0.1, name='Biases_adding')
            tf.histogram_summary(layerName + '/Biases', Biases)
        with tf.name_scope('Wx_b'):
            WxPlusB = tf.matmul(inputs, Weights) + Biases
        with tf.name_scope('Dropout'):
            WxPlusB = tf.nn.dropout(WxPlusB, keepProb)
        if actFunc is None:
            outputs = WxPlusB
        else:
            outputs = actFunc(WxPlusB)
        tf.histogram_summary(layerName + '/Outputs', outputs)
        return outputs
    
def calcAccuracy(groundTruthData, groundTruthLabel, keepProb):
    '''
    TODO: calculate the accuracy of prediction
    '''
    global outputFinal
    prediction = sess.run(outputFinal, feed_dict={xInputHandle: groundTruthData, dropoutKeepProb: keepProb})
    correctNum = tf.equal(tf.argmax(prediction, 1), tf.argmax(groundTruthLabel, 1))
    accuracy = tf.reduce_mean(tf.cast(correctNum, tf.float32))
    result = sess.run(accuracy, feed_dict={xInputHandle: groundTruthData, yInputHandle: groundTruthLabel, dropoutKeepProb: keepProb})
    return result

'''
Data preparation and feature extraction
'''
# xData = np.linspace(-1, 1, 500, dtype=np.float32)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, xData.shape).astype(np.float32)
# yData = np.square(xData) - 0.5 + noise
mnistRawData = input_data.read_data_sets('MNIST_data', one_hot=True)

# print(xData)
# print(yData)

'''
Model construction and training
'''
# Network parameters
# Structure parameters
inputNodeNum = 784
nodeNumHidLayer1 = 50
nodeNumHidLayer2 = 10
actFuncHidLayer1 = tf.nn.tanh
actFuncHidLayer2 = tf.nn.softmax
outputNodeNum = 10

# Input handles
dropoutKeepProb = tf.placeholder(tf.float32)    # keep probability of dropout
with tf.name_scope('inputs'):
    xInputHandle = tf.placeholder(tf.float32, shape=(None, inputNodeNum))
    yInputHandle = tf.placeholder(tf.float32, shape=(None, outputNodeNum))

# Structure: Layer by layer
outputHidLayer1 = addLayer('Hiden_Layer1', xInputHandle, inputNodeNum, nodeNumHidLayer1, dropoutKeepProb, actFunc=actFuncHidLayer1)
# outputHidLayer2 = addLayer('Hiden_Layer2', outputHidLayer1, nodeNumHidLayer1, nodeNumHidLayer2, actFunc=actFuncHidLayer2)
outputFinal = addLayer('Output_Layer', outputHidLayer1, nodeNumHidLayer1, outputNodeNum, dropoutKeepProb, actFunc=actFuncHidLayer2)

# Training parameters
learningRate = 0.5
batchSize = 100
trainingBatchNum = int(mnistRawData.train.num_examples / batchSize)
keepProb = 1
# Loss function
with tf.name_scope('Loss'):
    lossTensor = calcCrossEntropy1D(outputFinal, yInputHandle)
    # lossTensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputFinal, yInputHandle))
    tf.scalar_summary('Loss', lossTensor)
# Optimizer
with tf.name_scope('Training'):
    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(lossTensor)


'''
Tensorflow session running
'''
# Initialization
init = tf.initialize_all_variables()
hm_epoches = 1
# Merging
mergeForTBVisual = tf.merge_all_summaries()
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
            # Training data preparation
            batchXData, batchYData = mnistRawData.train.next_batch(batchSize)
            sess.run(trainStep, feed_dict={xInputHandle: batchXData, yInputHandle: batchYData, 
                    dropoutKeepProb: keepProb})
            if step % 50 == 0:
                train_rs = sess.run(mergeForTBVisual, 
                        feed_dict={xInputHandle: batchXData, yInputHandle: batchYData, dropoutKeepProb: 1})
                train_writer.add_summary(train_rs, step)
                test_rs = sess.run(mergeForTBVisual, 
                        feed_dict={xInputHandle: mnistRawData.test.images, yInputHandle: mnistRawData.test.labels, dropoutKeepProb: 1})
                test_writer.add_summary(test_rs, step)
                # testing for training data
                print('Epoch', epoch+1, 'training completed with the training accuracy of ', 
                        calcAccuracy(batchXData, batchYData, 1))
                # testing for testing data
                print('Epoch', epoch+1, 'training completed with the testing accuracy of ', 
                        calcAccuracy(mnistRawData.test.images, mnistRawData.test.labels, 1))

'''
Output and visualization
'''
