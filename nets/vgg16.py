# -*- encoding UTF-8 -*-
import os,math,time
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939,116.779,123.68]

class VGGNet:
    def __init__(self,data_dict):
        """创建网络结果，
           从模型中导入数据
        """
        self.data_dict = data_dict
    def get_covn_filter(self,name):
        return tf.constant(self.data_dict[name][0],name="conv")
    def get_fc_weight(self,name):
        return tf.constant(self.data_dict[name][0],name='fc')
    def get_bias(self,name):
        return tf.constant(self.data_dict[name][1],name='bias')
    def conv_layer(self,x,name):
        with tf.name_scope(name):
            w = self.get_covn_filter(name)
            b = self.get_bias(name)
            h = tf.nn.conv2d(x,w,[1,1,1,1],padding="SAME")
            h = tf.nn.bias_add(h,b)
            h = tf.nn.relu(h)
            return h
    def pooling_layer(self,x,name):
        return tf.nn.max_pool(
            x,
            [1,2,2,1],
            strides = [1,2,2,1],
            padding = "SAME",
            name = name
        )
    def fc_layer(self,x,name,activation=tf.nn.relu):
        with tf.name_scope(name):
            w = self.get_fc_weight(name)
            b = self.get_bias(name)
            h = tf.matmul(x,w)
            h = tf.nn.bias_add(h,b)
            if not activation is None:
                h = activation(h)
            return h
    def flatten_layer(self,x,name):
        with tf.name_scope(name):
            shape = x.get_shape().as_list()
            dim = 1
            for i in shape[1:]:
                dim*=i
            return tf.reshape(x,(-1,dim))
    def build(self,x_rgb):
        starttime = time.time()
        print("build model ...")
        r,g,b = tf.split(x_rgb,[1,1,1],axis=3)
        x_bgr = tf.concat([
            b-VGG_MEAN[0],
            g-VGG_MEAN[1],
            r-VGG_MEAN[2]
        ],axis = 3)
        assert x_bgr.get_shape().as_list()[1:]==[224,224,3]
        #开始创建网络
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')
        
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')
        
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')
        
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')
        
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')
        self.flatten5 = self.flatten_layer(self.pool5, 'flatten')
        '''
        self.fc6 = self.fc_layer(self.flatten5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        
        '''
        print("build model finished used %ds"%(time.time()-starttime))
        return self


        
    
    


