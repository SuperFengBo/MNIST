INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
import tensorflow as tf
def get_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return w
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
def forward(x):
    w1=get_weight([INPUT_NODE,LAYER1_NODE])
    b1=get_bias([LAYER1_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weight([LAYER1_NODE,OUTPUT_NODE])
    b2=get_bias(OUTPUT_NODE)
    y2=tf.matmul(y1,w2)+b2
    return y2