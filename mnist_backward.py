import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward

STEPS=50000
BATCH_SIZE=200
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.99
MOVING_AVERAGE_DECAY=0.99
regularizer=0.0001
def backward(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y=mnist_forward.forward(x,regularizer)
    
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem

    saver=tf.train.Saver()
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        #ckpt=tf.train.get_checkpoint_state('./model/')
        #if ckpt and ckpt.model_checkpoint_path:
            #saver.restore(sess,ckpt.model_checkpoint_path) 
        for i in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            result,loss_val,step,ce_val=sess.run([train_op,loss,global_step,ce],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print('After %d steps,loss is %f'%(step,loss_val))
                #print('learning_rate',learning_rate_val)
                saver.save(sess,'./model/',global_step=global_step)
            
mnist=input_data.read_data_sets("./data/",one_hot=True)
backward(mnist)

