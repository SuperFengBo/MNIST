IMAGE_SIZE=28
IMAGE_CHANNEL=1
OUTPUT_NODE=10
import tensorflow as tf
import numpy as np
CONV1_SIZE=3
CONV1_CHANNEL=1
CONV1_KERNEL_NUM=2
CONV2_SIZE=5
CONV2_CHANNEL=CONV1_KERNEL_NUM
CONV2_KERNEL_NUM=2
regularizer=0.0001
def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:
        tf.contrib.layers.l2_regularizer(regularizer)(w)
    return w
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
def forward(x,regularizer):
    conv1_w=get_weight([CONV1_SIZE,CONV1_SIZE,CONV1_CHANNEL,CONV1_KERNEL_NUM],regularizer)
    conv1_b=get_bias([CONV1_KERNEL_NUM])
    y1=tf.nn.conv2d(x,conv1_w,[1,1,1,1],padding='SAME')+conv1_b
    print('***************y1',y1)
    conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV2_CHANNEL,CONV2_KERNEL_NUM],regularizer)
    conv2_b=get_bias([CONV2_KERNEL_NUM])
    y2=tf.nn.conv2d(y1,conv2_w,[1,1,1,1],padding='SAME')+conv2_b    
    
    y=tf.reshape(y2,[-1,IMAGE_SIZE*IMAGE_SIZE*CONV2_KERNEL_NUM])    
    
    fc1_w=get_weight([IMAGE_SIZE*IMAGE_SIZE*CONV2_KERNEL_NUM,OUTPUT_NODE],regularizer)
    fc1_b=get_bias([OUTPUT_NODE])
    y_fc=tf.matmul(y,fc1_w)+fc1_b
    return y_fc
