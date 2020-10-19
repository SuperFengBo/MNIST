import tensorflow as tf
import cifar_forward
import _pickle as pickle
import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
STEPS=1000000
BATCH_SIZE=1000
LEARNING_RATE_BASE=0.000005
LEARNING_RATE_DECAY=0.9999
MOVING_AVERAGE_DECAY=0.99
logdir='./log/'
MODEL_DIR='./cifar_model/'
cifar_path='./cifar-10-batches-py'

#读取文件
def load_pickle(f):
    version = platform.python_version_tuple() # 取python版本号
    if version[0] == '2':
        return  pickle.load(f) # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)   # dict类型
    X = datadict['data']        # X, ndarray, 像素值
    Y = datadict['labels']      # Y, list, 标签, 分类
    
    # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
    # transpose，转置
    # astype，复制，同时指定类型
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y
def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = [] # list
  ys = []
  
  # 训练集batch 1～5
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X) # 在list尾部添加对象X, x = [..., [X]]
    ys.append(Y)    
  Xtr = np.concatenate(xs) # [ndarray, ndarray] 合并为一个ndarray
  Ytr = np.concatenate(ys)
  del X, Y

  # 测试集
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
def backward(cifar_path):
    x=tf.placeholder(tf.float32,[None,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_CHANNEL])
    y_=tf.placeholder(tf.float32,[None,cifar_forward.OUTPUT_NODE])
    y=cifar_forward.forward(x,1,cifar_forward.regularizer)
    
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem
    tf.summary.scalar('loss',loss)
    saver=tf.train.Saver()
    global_step=tf.Variable(0,trainable=False)
    Xtr, Ytr, Xte, Yte=load_CIFAR10(cifar_path)
    xs,ys=Xtr,Ytr
    xs_batch=np.multiply(xs,1.0/255.0)
    #ys_batch=np.eye(10)[ys]
    ys_batch=tf.one_hot(ys,10)
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        np.shape(Ytr)[0]/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )  
    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    summary_merged=tf.summary.merge_all()
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')      

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options) ) as sess:
        sess.run(tf.global_variables_initializer()) 
        ckpt=tf.train.get_checkpoint_state(MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path) 
        for i in range(STEPS):
            for j in range(int(50000/BATCH_SIZE)):
                ys_batch_val=sess.run(ys_batch)
                xs,ys=xs_batch[BATCH_SIZE*j:BATCH_SIZE*(j+1)],ys_batch_val[BATCH_SIZE*j:BATCH_SIZE*(j+1)]                       
                learning_rate_val,summary_,result,loss_val,step,ce_val=sess.run([learning_rate,summary_merged,train_op,loss,global_step,ce],feed_dict={x:xs,y_:ys})           
                if step%(50000/BATCH_SIZE)==0:
                    print('After %d steps,loss is %f'%(step,loss_val))
                    saver.save(sess,MODEL_DIR,global_step=global_step)            
                    writer=tf.summary.FileWriter(logdir)
                    writer.add_summary(summary_,global_step=step)
                    print('learning_rate',learning_rate_val)
backward(cifar_path)
#print(device_lib.list_local_devices())

