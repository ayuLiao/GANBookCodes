import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
# 读入mnist数据，没有则自动从互联网上下载
mnist = input_data.read_data_sets('./data/MNIST_data')


# img = mnist.train.images[500]
# 以灰度图的形式读入
# plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
# plt.show()

def get_inputs(real_size, noise_size):
    '''
    真实图像tensor与噪声图像tensor
    :param real_size:
    :param noise_size:
    :return:
    '''
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

    return real_img, noise_img

def generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    '''
    生成器
    :param noise_img: 生成器生成的噪声图片
    :param n_units: 隐藏层单元数
    :param out_dim: 生成器输出的tensor的size，应该是32*32=784
    :param reuse: 是否重用空间
    :param alpha: leakey ReLU系数
    :return:
    '''

    with tf.variable_scope("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise_img, n_units) #隐层
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs

def discirminator(img, n_units, reuse=False, alpha=0.01):
    '''
    判别器
    :param img: 图片（真实图片/生成图片）
    :param n_units:
    :param reuse:
    :param alpha:
    :return:
    '''

    with tf.variable_scope('discriminator', reuse=reuse):
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs

# train_data = mnist.train.images  # Returns np.array
img_size = mnist.train.images[0].shape[0]#真实图片大小
noise_size = 100 #噪声,Generator的初始输入
g_units = 128#生成器隐层参数
d_units = 128
alpha = 0.01 #leaky ReLU参数
learning_rate = 0.001 #学习速率
smooth = 0.1 #标签平滑

tf.reset_default_graph()
#读入
real_img, noise_img = get_inputs(img_size,noise_size)

#生成器
g_logits, g_outputs = generator(noise_img, g_units, img_size)

#判别器
d_logits_real, d_outputs_real = discirminator(real_img, d_units)
# 传入生成图片，为其打分
d_logits_fake, d_outputs_fake = discirminator(g_outputs, d_units, reuse=True)


d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_real, labels=tf.ones_like(d_logits_real))*(1-smooth)) #真实图片是真实的，要打1分，其差距就是损失

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake))) #生成图片是虚假的，要打0分，其差距就是损失

d_loss = tf.add(d_loss_real, d_loss_fake)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_fake, labels=tf.ones_like(d_logits_fake))*(1-smooth))

train_vars = tf.trainable_variables()

# generator中的tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator中的tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars) #最小化判别模型的损失
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars) #最小化生成模型的损失

batch_size = 64
epochs = 500 #训练迭代轮数
n_sample = 25 #抽取样本数

samples = [] #存储测试样例
losses = [] #存储loss
#保存生成器变量
saver = tf.train.Saver(var_list=g_vars)

#开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images * 2 -1
            #生成噪音图片
            batch_noise = np.random.uniform(-1,1,size=(batch_size, noise_size))

            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img:batch_noise}) #训练判别模型
            _ = sess.run(g_train_opt, feed_dict={noise_img:batch_noise}) #训练生成模型

        #每一轮训练完后，都计算一下loss
        train_loss_d = sess.run(d_loss, feed_dict={real_img:batch_images, noise_img:batch_noise})
        # real img loss
        train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img:batch_images, noise_img:batch_noise})
        # fake img loss
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img:batch_images, noise_img:batch_noise})
        # generator loss
        train_loss_g = sess.run(g_loss, feed_dict= {noise_img: batch_noise})

        print("训练轮数 {}/{}...".format(e + 1, epochs),
              "判别器总损失: {:.4f}(真实图片损失: {:.4f} + 虚假图片损失: {:.4f})...".format(train_loss_d, train_loss_d_real,
                                                                                  train_loss_d_fake),"生成器损失: {:.4f}".format(train_loss_g))
        # 记录各类loss值
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        # 抽取样本后期进行观察
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img:sample_noise})
        samples.append(gen_samples)
        # 存储checkpoints
        saver.save(sess, './data/generator.ckpt')

with open('./data/train_samples.pkl', 'wb') as f:
    pickle.dump(samples,f)

figfig, axax  =  plt.subplots(figsize=(20,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator Total Loss')
plt.plot(losses.T[1], label='Discriminator Real Loss')
plt.plot(losses.T[2], label='Discriminator Fake Loss')
plt.plot(losses.T[3], label='Generator')
plt.title("Training Losses")
plt.legend()




