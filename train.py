# -*- encoding UTF-8 -*-
import numpy as np
import os
from nets.vgg16 import VGGNet
import imageio
import tensorflow as tf
class train():
    def __init__(self):
        self.D_ROOT = os.path.abspath(os.path.dirname(__file__))
        self.D_DATA = os.path.join(self.D_ROOT,"data")
        self.D_IMG_OUT = os.path.join(self.D_ROOT,"run_time")
        self.F_VGG16 = os.path.join(self.D_DATA,"vgg16.npy") #VGG16模型
        self.F_STYLE_IMG = os.path.join(self.D_DATA,"xingkong.jpeg") #风格图地址
        self.F_CONTENT_IMG = os.path.join(self.D_DATA,"gugong.jpg") #内容图地址
        #创建图像
        self.IMG_STYLE = self.__load_img(self.F_STYLE_IMG) #读取风格图
        self.IMG_CONTENT = self.__load_img(self.F_CONTENT_IMG)#读取内容图
        self.IMG_RESULT = self.__img_result((1,224,224,3),127.5,20)
        #占位符
        self.INPUT_STYLE = tf.placeholder(tf.float32,shape=(1,224,224,3))
        self.INPUT_CONTENT = tf.placeholder(tf.float32,shape=(1,224,224,3))
        self.__compile_net()
        #定义损失/训练
        self.LAMBDA_C = 0.1
        self.LAMBDA_S = 500
        self.ECOPH = 100
        self.LEARNING_RATE = 10
       

    def get_vgg_data(self):
        if not hasattr(self,"VGG_DATA"):
            self.VGG_DATA = np.load(self.F_VGG16,encoding='latin1').item()
        return self.VGG_DATA
    def __load_img(self,image_dir):
        img = imageio.imread(image_dir)
        img_array = np.array(img)
        return np.asarray([img_array])
    def __img_result(self,shape,mean,stddev):
        result = tf.truncated_normal(shape,mean=mean,stddev=stddev,dtype=tf.float32)
        return tf.Variable(result)
    def __compile_net(self):
        VGG_DATA = self.get_vgg_data()
        self.STYLE_NET = VGGNet(VGG_DATA).build(self.INPUT_STYLE)
        self.CONTENT_NET = VGGNet(VGG_DATA).build(self.INPUT_CONTENT)
        self.RESULT_NET = VGGNet(VGG_DATA).build(self.IMG_RESULT)
        return self
    def __gram_matrix(self,x):
        b,w,h,c = x.get_shape().as_list()
        F = tf.reshape(x,[b,w*h,c])
        G = tf.matmul(F,F,adjoint_a=True)
        return G/tf.constant(w*h*c,dtype=tf.float32)
    def style_loss(self,style_features,result_features):
        style_loss = tf.zeros([1])
        style_gram_list = [self.__gram_matrix(item) for item in style_features]
        result_gram_list =[self.__gram_matrix(item) for item in result_features]
        for sg,rg in zip(style_gram_list,result_gram_list):
            style_loss += tf.reduce_mean(tf.square(sg-rg),[1,2])
        return style_loss
    def content_loss(self,content_features,result_features):
        content_loss = tf.zeros([1])
        for cf,rf in zip(content_features,result_features):
            content_loss += tf.reduce_mean(tf.square(cf-rf),[1,2,3])
        return content_loss
    def loss(self):
        style_features = [
            #self.STYLE_NET.conv1_2,
            self.STYLE_NET.conv4_3,
        ]
        content_features = [
            self.CONTENT_NET.conv1_2,
            #self.CONTENT_NET.conv4_3
        ]
        result_style_features = [
            #self.RESULT_NET.conv1_2,
            self.RESULT_NET.conv4_3
        ]
        result_content_features = [
            self.RESULT_NET.conv1_2,
            #self.RESULT_NET.conv4_3
        ]
        content_loss = self.content_loss(content_features,result_content_features)
        style_loss = self.style_loss(style_features,result_style_features)
        loss = self.LAMBDA_C*content_loss+self.LAMBDA_S*style_loss
        return loss,content_loss,style_loss
    def __out_put(self,sess,step):
        img = self.IMG_RESULT.eval(sess)
        img = np.clip(img[0],0,255)
        img_arr = np.asarray(img,dtype=np.uint8)
        if not os.path.isdir(self.D_IMG_OUT):
            os.makedirs(self.D_IMG_OUT)
        im_name = os.path.join(self.D_IMG_OUT,"%d.png"%step)
        imageio.imsave(im_name,img_arr)
    def run(self):
        loss,content_loss,style_loss = self.loss() 
        train_option = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.ECOPH):
                loss_va,c_loss,s_loss,_=sess.run([loss,content_loss,style_loss,train_option],feed_dict={
                    self.INPUT_CONTENT:self.IMG_CONTENT,
                    self.INPUT_STYLE:self.IMG_STYLE,
                })
                msg = "step=%02d,loss=%.2f,c_loss=%.2f,s_loss=%.2f"%(i+1,loss_va[0],c_loss[0],s_loss[0])
                print(msg)
                self.__out_put(sess,i+1)


        
if __name__ == "__main__":
    tModel = train()
    tModel.run()



