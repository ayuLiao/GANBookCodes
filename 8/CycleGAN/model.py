# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random


from layers import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width


batch_size = 1
pool_size = 50
ngf = 32
ndf = 64



#残差网络，通过卷积层抽取部分特征+直接传递部分特征，让生成器生成的内容与输入内容不会有太大的差距
def resnet_block(input, dim, name="resnet"):
    
    with tf.variable_scope(name):

        out_res = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
        return tf.nn.relu(out_res + input)

def build_generator_resnet_6blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        x = conv(pad_input, ngf, f, f, 1, 1, 0.02,name="conv_1")
        x = conv(x, ngf*2, ks, ks, 2, 2, 0.02,"SAME","conv_2")
        x = conv(x, ngf*4, ks, ks, 2, 2, 0.02,"SAME","conv_3")

        for i in range(1,7):
            x = resnet_block(x, ngf*4, 'resnet_'+str(i))

        x = deconv(x, [batch_size,64,64,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","conv_4")
        x = deconv(x, [batch_size,128,128,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","conv_5")
        x = tf.pad(x,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        x = conv(x, img_layer, f, f, 1, 1, 0.02,"VALID","conv_6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(x,"t1")


        return out_gen

def generator(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        x = conv(pad_input, ngf, f, f, 1, 1, 0.02,name="conv_1")
        x = conv(x, ngf*2, ks, ks, 2, 2, 0.02,"SAME","conv_2")
        x = conv(x, ngf*4, ks, ks, 2, 2, 0.02,"SAME","conv_3")

        for i in range(1,10):
            x = resnet_block(x, ngf * 4, "resnet_"+str(i))

        x = deconv(x, [batch_size,128,128,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","conv_4")
        x = deconv(x, [batch_size,256,256,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","conv_5")
        x = conv(x, img_layer, f, f, 1, 1, 0.02,"SAME","conv_6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(x,"tanh_1")


        return out_gen


def discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        x = conv(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "conv_1", do_norm=False, relufactor=0.2)
        x = conv(x, ndf*2, f, f, 2, 2, 0.02, "SAME", "conv_2", relufactor=0.2)
        x = conv(x, ndf*4, f, f, 2, 2, 0.02, "SAME", "conv_3", relufactor=0.2)
        x = conv(x, ndf*8, f, f, 1, 1, 0.02, "SAME", "conv_4",relufactor=0.2)
        x = conv(x, 1, f, f, 1, 1, 0.02, "SAME", "conv_5",do_norm=False,do_relu=False)

        return x


def patch_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = conv(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "conv_1", do_norm="False", relufactor=0.2)
        o_c2 = conv(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "conv_2", relufactor=0.2)
        o_c3 = conv(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "conv_3", relufactor=0.2)
        o_c4 = conv(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "conv_4", relufactor=0.2)
        o_c5 = conv(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "conv_5",do_norm=False,do_relu=False)

        return o_c5