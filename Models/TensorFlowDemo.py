# TensorFlow Demo
import tensorflow as tf
import numpy as np
# validation
hello=tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello))

# linear fitting
# create data
xData = np.random.rand(100).astype(np.float32)
yData = xData * 0.1 + 0.3
### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Biases = tf.Variable(tf.zeros([1]))
yPre = xData * Weights + Biases
# learning parameters
loss = tf.reduce_mean(tf.square(yPre-yData))
learningRate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learningRate)
train = optimizer.minimize(loss)
### create tensorflow structure end

# Activeation and training
init = tf.global_variables_initializer()
# session
sess = tf.Session()
sess.run(init)
# run training
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Biases))

# matrix multiplying
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1, matrix2)
# Activation
# Method 1
sess = tf.Session()
result = sess.run(product)
print('Method 1:', result)
sess.close()

# Method 2
with tf.Session() as sess:
    result = sess.run(product)
    print('Method 2:', result)

# Counter
state = tf.Variable(0, name = 'counter')
# one = tf.constant(1)
# newState = tf.add(state, one)
# update = tf.assign(state, newState)
timeTot = tf.Variable(3)
# initialization
init = tf.global_variables_initializer()
# activation
with tf.Session() as sess:
    sess.run(init)
    for step in range (sess.run(timeTot)):
        # result = sess.run(update)
        # state = state + 1
        print(sess.run(state))
        print(type(state))