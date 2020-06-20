import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

# the implements of leakyRelu
def lrelu(x , alpha=0.2 , name="LeakyReLU"):
    with tf.name_scope(name):
        return tf.maximum(x , alpha*x)


'''
初始化权重时，采用He initializer的方式进行初始化，He initializer方式有助于ReLU训练
'''
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    print("current", shape[:-1], fan_in)
    std = gain / np.sqrt(fan_in) # He init

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, gain=np.sqrt(2), use_wscale=False, padding='SAME',
           name="conv2d", with_w=False):
    with tf.variable_scope(name):

        w = get_weight([k_h, k_w, input_.shape[-1].value, output_dim], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, input_.dtype)

        if padding == 'Other':
            padding = 'VALID'
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases

        else:
            return conv

def fully_connect(input_, output_size, gain=np.sqrt(2), use_wscale=False, name=None, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name or "Linear"):

    w = get_weight([shape[1], output_size], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, input_.dtype)
    bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

    output = tf.matmul(input_, w) + bias

    if with_w:
        return output, with_w, bias

    else:
        return output

def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse= reuse , updates_collections=None)

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def downscale2d(x, k=2):
    # avgpool wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

def Pixl_Norm(x, epsilon=1e-8):
    '''
    PG-GAN 抛弃BN层、IN层，而使用自己提出的Pixl Norm
    Pixl Norm 用于约束生成器与判别器不健康竞争造成的信号范围越界问题，
    提出了He's init动态初始化来平衡学习率。
    :param x:
    :param epsilon:
    :return:
    '''
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    with tf.variable_scope('PixelNorm'):
        # LRN 局部归一化，侧抑制
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keep_dims=True) + epsilon)

def MinibatchstateConcat(input, averaging='all'):
    '''
    MSD --> 减缓GAN模式崩溃现象
    :param input:
    :param averaging:
    :return:
    '''
    s = input.shape
    # 多样性度量
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")
    # tf.tile铺平给定的张量
    vals = tf.tile(vals, multiples=[s[0], s[1], s[2], 1])
    #连接输入与多样性度量
    return tf.concat([input, vals], axis=3)











