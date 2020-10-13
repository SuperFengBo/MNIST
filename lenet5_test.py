import tensorflow as tf
import mnist_forward
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import lenet5_forward
BATCH_SIZE=5000
mnist=input_data.read_data_sets("./data/",one_hot=True)
def test(mnist):
    with tf.Graph().as_default() as g:

        x=tf.placeholder(tf.float32,[None,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_CHANNEL])
        y_=tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
        y=lenet5_forward.forward(x,lenet5_forward.regularizer)

        correct_pre=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1)) 
        correct_pre=tf.cast(correct_pre,tf.float32)
        acc=tf.reduce_mean(correct_pre)
        saver=tf.train.Saver()
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state('./lenet5_model/')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    xs,ys=mnist.test.next_batch(BATCH_SIZE)
                    xs=np.reshape(xs,[-1,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_CHANNEL])
                    acc_val=sess.run(acc,feed_dict={x:xs,y_:ys}) 
                    print('acc is',acc_val )
            time.sleep(5)
test(mnist)


