import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
import time
from datetime import timedelta
from sklearn import model_selection

def curtime():
    return time.asctime(time.localtime(time.time()))


print curtime()+" Program begin "


sess = tf.InteractiveSession()


BigTrain = 0

if BigTrain == 0:    
    X_total = np.load('../train_X.npy')
    y_total_orz = np.load('../train_y.npy')
else:
    X_total = np.load('../new_X.npy')
    y_total_orz = np.load('../new_y.npy')
    
print "Total data:",  X_total.shape, y_total_orz.shape

total_size = X_total.shape[0]
test_size = 2000
train_size = total_size - test_size

batch_size = 50
epochs = 100

reg_para = 0.0001

#X_total = X_total[:total_size,:]
X_total = X_total.astype(np.float32)/256
#y_total_orz = y_total_orz[:total_size]

y_total = np.zeros((total_size,10))
for i in xrange(total_size):
    label = y_total_orz[i]
    if label == 10:
        y_total[i,0] = 1
    else:
        y_total[i,label] = 1

#X_train = X_total[:train_size,:]
#y_train = y_total[:train_size,:]
#X_test = X_total[train_size:total_size,:]
#y_test = y_total[train_size:total_size,:]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X_total,y_total,test_size = test_size)
assert X_train.shape[0]==train_size
assert y_train.shape[0]==train_size
print "Training data:", X_train.shape, y_train.shape
print "Test data:",X_test.shape, y_test.shape
#plt.imshow(X_train[6])
#print y_train[6]
#plt.imshow(X_test[8])
#print y_test[8]
        

x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,32,32,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3,3,64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = avg_pool_2x2(h_conv3)

W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + \
    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_conv3)

loss = cross_entropy + reg_para * regularizers

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


print curtime()+" Training begin "
begintime = time.time()  
total_steps = int(epochs*train_size/batch_size)
for i in range(total_steps):
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]
    if i%100 == 0:
        used_time = str(timedelta(seconds=int(time.time()-begintime)))
        train_accuracy = accuracy.eval(feed_dict={
                x:X_batch, y_: y_batch, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={
                x:X_test, y_: y_test, keep_prob: 1.0})
        print("step %d/%d, used time %s, training accuracy %g, test accuracy %s" %(i,totol_steps, used_time, train_accuracy, test_accuracy))
        print "regularization:",reg_para*regularizers.eval(),"loss:",loss.eval(feed_dict={
                x:X_batch, y_: y_batch, keep_prob: 0.5})
    train_step.run(feed_dict={x: X_batch, y_: y_batch, keep_prob: 0.5})

print curtime()+"Training complete "


test_X = np.load('../test_X.npy')
#test_X = test_X[:10]
test_m = test_X.shape[0]
print curtime()+"Prediction begin "
prediction = (tf.argmax(y_conv,1)).eval(feed_dict={x:test_X, keep_prob:1.0})
prediction[prediction==0]=10
dfout = pd.DataFrame()
dfout['ImageId']= np.arange(test_m)
dfout['label']=prediction
dfout.to_csv('../result0.csv',index=False)


print curtime()+"Program end "
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))










