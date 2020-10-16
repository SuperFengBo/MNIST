import tensorflow as tf
import cifar_forward

import time
import numpy as np
import cifar_forward
import os
import platform
import _pickle as pickle
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
MODEL_DIR='./cifar_model/'
train=False
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
  Xtr, Ytr, Xte, Yte=load_CIFAR10(cifar_path)
  xs,ys=Xte,Yte
  xs_test_batch=np.multiply(xs,1.0/255.0)
  ys_test_batch=np.eye(10)[ys]
def test(cifar_path):
    with tf.Graph().as_default() as g:

        x=tf.placeholder(tf.float32,[None,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_CHANNEL])
        y_=tf.placeholder(tf.float32,[None,cifar_forward.OUTPUT_NODE])
        y=cifar_forward.forward(x,train,cifar_forward.regularizer)

        correct_pre=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1)) 
        correct_pre=tf.cast(correct_pre,tf.float32)
        acc=tf.reduce_mean(correct_pre)
        saver=tf.train.Saver()

        _, _, Xte, Yte=load_CIFAR10(cifar_path)
        xs,ys=Xte,Yte
        xs_test_batch=np.multiply(xs,1.0/255.0)
        ys_test_batch=np.eye(10)[ys]        
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(MODEL_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    #xs=np.reshape(xs,[-1,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_CHANNEL])
                    acc_val=sess.run(acc,feed_dict={x:xs_test_batch[0:1000],y_:ys_test_batch[0:1000]}) 
                    print('acc is',acc_val )
            time.sleep(5)
test(cifar_path)

