import tensorflow as tf
import mnist_forward
from tensorflow.examples.tutorials.mnist import input_data
import time
BATCH_SIZE=5000
mnist=input_data.read_data_sets("./data/",one_hot=True)
def test(mnist):
    with tf.Graph().as_default() as g:

        x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y_=tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
        y=mnist_forward.forward(x)

        correct_pre=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1)) 
        correct_pre=tf.cast(correct_pre,tf.float32)
        acc=tf.reduce_mean(correct_pre)
        saver=tf.train.Saver()
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state('./model/')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    xs,ys=mnist.test.next_batch(BATCH_SIZE)
                    acc_val=sess.run(acc,feed_dict={x:xs,y_:ys}) 
                    print('acc is',acc_val )
            time.sleep(5)
test(mnist)


