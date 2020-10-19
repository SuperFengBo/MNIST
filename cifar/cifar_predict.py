import tensorflow as tf
import cifar_forward
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
regularizer=0.0001
def prepic(image):
    img = Image.open(image)
    reim=img.resize((32,32),Image.ANTIALIAS)
    #im_arr=np.array(reim.convert("L"))    
    # threshold=50
    # for i in range(32):
    #     for j in range(32):
    #         im_arr[i][j] = 255 - im_arr[i][j]
    #         if (im_arr[i][j] < threshold):
    #             im_arr[i][j] = 0
    #         else: im_arr[i][j] = 255  
    plt.imshow(reim)
    #plt.show()  
    nm_arr=np.reshape(reim,[1,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_CHANNEL])
    nm_arr=nm_arr.astype(np.float32)
    im_ready=np.multiply(nm_arr,1.0/255.0)
    return im_ready
def predict(im_ready):
    with tf.Graph().as_default() as p:  
        x=tf.placeholder(tf.float32,[None,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_SIZE,cifar_forward.IMAGE_CHANNEL])
        y=cifar_forward.forward(x,None,regularizer)
        predict_op=tf.arg_max(y,1)
        saver=tf.train.Saver()
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state('./cifar_model/')
            saver.restore(sess,ckpt.model_checkpoint_path)
            result=sess.run(predict_op,feed_dict={x:im_ready})
            print('the prediction is %d '%(result),end="")
    return result

#predict(image)
def eachFile(filepath):
    pathDir=os.listdir(filepath)
    correct_num=0
    for allDir in pathDir:
        impath=os.path.join('%s%s'%(filepath,allDir))
        answer=impath[-5:-4]
        result=predict(prepic(impath))
        #print('child',child[-5:-4])
        if int(answer)==result:
            print("√")
            correct_num = correct_num + 1
        else:
            print("X",'the correct answer is',answer)
            plt.show()
        print('correct_num',correct_num)
# while True:
#     impath,answer=eachFile(./mnist_test_jpg_10000/)
#     print('input is %d '%(i),end="")
#     result=predict(prepic('./mnist_test_jpg_10000/%d.png'%(i)))
#     if result[0]==i:
#         print("√")
#     else:
#         print("X")
eachFile('./pic/')
#eachFile('./mnist_test_jpg_10000/')