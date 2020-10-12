import tensorflow as tf
import mnist_forward
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def prepic(image):
    img = Image.open(image)
    reim=img.resize((28,28),Image.ANTIALIAS)
    reim=np.array(reim.convert("L"))
    plt.imshow(reim)
    plt.show()
    nm_arr=reim.reshape([1,784])
    nm_arr=nm_arr.astype(np.float32)
    im_ready=np.multiply(nm_arr,1.0/255.0)
    return im_ready
def predict(im_ready):
    with tf.Graph().as_default() as p:  
        x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y=mnist_forward.forward(x)
        predict_op=tf.arg_max(y,1)
        saver=tf.train.Saver()
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state('./model/')
            saver.restore(sess,ckpt.model_checkpoint_path)
            result=sess.run(predict_op,feed_dict={x:im_ready})
            print('the prediction is %d '%(result),end="")
    return result

#predict(image)
for i in range(10):
    print('input is %d '%(i),end="")
    result=predict(prepic('./num/%d.png'%(i)))
    if result[0]==i:
        print("√")
    else:
        print("X")