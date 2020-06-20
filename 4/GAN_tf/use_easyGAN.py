import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('./data/train_samples.pkl', 'rb') as f:
	samples = pickle.load(f)

def view_img(epoch,samples):
	fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharex=True,sharey=True)
	for ax,img in zip(axes.flatten(), samples[epoch][1]):
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		im = ax.imshow(img.reshape((28,28)), cmap="Greys_r")
	plt.show()

view_img(-1,samples)

def view_all():
	#从0开始抽取，每次隔50
	epoch_index = [x for x in range(0,500,50)]
	show_imgs = []
	for i in epoch_index:
		show_imgs.append(samples[i][1])

	rows, cols = len(epoch_index) ,len(samples[0][1])
	fig, axes = plt.subplots(figsize=(30,20), nrows=rows, ncols=cols, sharex=True,sharey=True)

	index = range(0, 500, int(500/rows))

	for sample, ax_row in zip(show_imgs, axes):
		for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
			ax.imshow(img.reshape((28,28)), cmap='Greys_r')
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

	plt.show()

import easyGAN_MNIST

# 定义参数# 定义参数
 # 真实图像的size# 真实图像的s
img_size = easyGAN_MNIST.mnist.train.images[0].shape[0]
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1
noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

# 加载我们的生成器变量
saver = tf.train.Saver(var_list=easyGAN_MNIST.g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
    gen_samples = sess.run(easyGAN_MNIST.generator(noise_img, g_units, img_size, reuse=True),
                           feed_dict={noise_img: sample_noise})

_ = view_img(0, [gen_samples])