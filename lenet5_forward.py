IMAGE_SIZE=28
IMAGE_CHANNEL=1
OUTPUT_NODE=10
import tensorflow as tf
import numpy as np
CONV1_SIZE=5
CONV1_CHANNEL=1
CONV1_KERNEL_NUM=32
CONV2_SIZE=5
CONV2_CHANNEL=CONV1_KERNEL_NUM
CONV2_KERNEL_NUM=64
regularizer=0.0001
FC1_SIZE=512
def get_weight(shape,regulaizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:
        tf.contrib.layers.l2_regularizer(regularizer)(w)
    return w
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
def forward(x,train,regularizer):
    conv1_w=get_weight([CONV1_SIZE,CONV1_SIZE,IMAGE_CHANNEL,CONV1_KERNEL_NUM],regularizer)
    conv1_b=get_bias([CONV1_KERNEL_NUM])
    conv1=tf.nn.conv2d(x,conv1_w,[1,1,1,1],padding='SAME')
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1=tf.nn.max_pool(relu1,[1,2,2,1],[1,2,2,1],padding='SAME')

    conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    conv2_b=get_bias([CONV2_KERNEL_NUM])
    conv2=tf.nn.conv2d(pool1,conv2_w,[1,1,1,1],padding='SAME')  
    relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2=tf.nn.max_pool(relu2,[1,2,2,1],[1,2,2,1],padding='SAME')
    pool_shape=pool2.get_shape().as_list()

    y_reshape=tf.reshape(pool2,[-1,pool_shape[1]*pool_shape[2]*pool_shape[3]])    
    
    fc1_w=get_weight([pool_shape[1]*pool_shape[2]*pool_shape[3],FC1_SIZE],regularizer)
    fc1_b=get_bias(FC1_SIZE)
    fc1=tf.matmul(y_reshape,fc1_w)+fc1_b
    if train: 
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w=get_weight([FC1_SIZE,OUTPUT_NODE],regularizer)
    fc2_b=get_bias([OUTPUT_NODE])
    fc2=tf.matmul(fc1,fc2_w)+fc2_b   

    return fc2
