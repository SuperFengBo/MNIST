import tensorflow as tf
import lenet5_forward
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
regularizer=0.0001
def prepic(image):
    img = Image.open(image)
    reim=img.resize((28,28),Image.ANTIALIAS)
    im_arr=np.array(reim.convert("L"))    
    # threshold=50
    # for i in range(28):
    #     for j in range(28):
    #         im_arr[i][j] = 255 - im_arr[i][j]
    #         if (im_arr[i][j] < threshold):
    #             im_arr[i][j] = 0
    #         else: im_arr[i][j] = 255  
    plt.imshow(im_arr)
    #plt.show()  
    nm_arr=im_arr.reshape([1,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_CHANNEL])
    nm_arr=nm_arr.astype(np.float32)
    im_ready=np.multiply(nm_arr,1.0/255.0)
    return im_ready
def predict(im_ready):
    with tf.Graph().as_default() as p:  
        x=tf.placeholder(tf.float32,[None,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_SIZE,lenet5_forward.IMAGE_CHANNEL])
        y=lenet5_forward.forward(x,regularizer)
        predict_op=tf.arg_max(y,1)
        saver=tf.train.Saver()
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state('./lenet5_model/')
            saver.restore(sess,ckpt.model_checkpoint_path)
            result=sess.run(predict_op,feed_dict={x:im_ready})
            print('the prediction is %d '%(result),end="")
    return result

#predict(image)
def eachFile(filepath):
    pathDir=os.listdir(filepath)
    for allDir in pathDir:
        impath=os.path.join('%s%s'%(filepath,allDir))
        answer=impath[-5:-4]
        result=predict(prepic(impath))
        #print('child',child[-5:-4])
        if int(answer)==result:
            print("√")
        else:
            print("X",'the correct answer is',answer)
            plt.show()
# while True:
#     impath,answer=eachFile(./mnist_test_jpg_10000/)
#     print('input is %d '%(i),end="")
#     result=predict(prepic('./mnist_test_jpg_10000/%d.png'%(i)))
#     if result[0]==i:
#         print("√")
#     else:
#         print("X")
eachFile('./mnist_test_jpg_10000/')
