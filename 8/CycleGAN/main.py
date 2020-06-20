import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random
import sys
from glob import glob

from layers import *
from model import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./checkpoint/"

# trainA_path = r"/Users/ayuliao/Desktop/GAN/data/CycleGANData/horse2zebra/trainA/*.jpg"
# trainB_path = r"/Users/ayuliao/Desktop/GAN/data/CycleGANData/horse2zebra/trainB/*.jpg"
trainA_path =r"/input/horse2zebra-1/trainA/*.jpg"
trainB_path =r"/input/horse2zebra-1/trainB/*.jpg"
temp_check = 0



max_epoch = 1
max_images = min(len(glob(trainA_path)), len(glob(trainB_path)))

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class CycleGAN():

    def input_setup(self):

        ''' 
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''
        # match_filenames_once 
        filenames_A = tf.train.match_filenames_once(trainA_path)
        # filenames_A = tf.train.match_filenames_once(r"/input/horse2zebra-1/trainA/*.jpg") #RussellClound
        self.queue_length_A = tf.size(filenames_A)

        filenames_B = tf.train.match_filenames_once(trainB_path)
        self.queue_length_B = tf.size(filenames_B)
        
        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)

    

    def input_read(self, sess):


        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
         self.fake_images_A / self.fake_images_B 
         self.A_input / self.B_input
        '''

        # Loading images into the tensors 
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))


        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

        for i in range(max_images): 
            image_tensor = sess.run(self.image_A)

            if(image_tensor.size == img_size*batch_size*img_layer):
                self.A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            if(image_tensor.size == img_size*batch_size*img_layer):
                self.B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))


        coord.request_stop()
        coord.join(threads)




    def model_setup(self):

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        
        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            #A --> B'
            self.fake_B = generator(self.input_A, name="g_AB")
            #B --> A'
            self.fake_A = generator(self.input_B, name="g_BA")
            self.rec_A = discriminator(self.input_A, "d_A")
            self.rec_B = discriminator(self.input_B, "d_B")

            scope.reuse_variables() 

            self.fake_rec_A = discriminator(self.fake_A, "d_A")
            self.fake_rec_B = discriminator(self.fake_B, "d_B")
            # B' --> A_cyc
            self.cyc_A = generator(self.fake_B, "g_BA")
            # A' --> B_cyc
            self.cyc_B = generator(self.fake_A, "g_AB")

            scope.reuse_variables()

            self.fake_pool_rec_B = discriminator(self.fake_pool_B, "d_B")



    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        self.disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        self.disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        self.g_loss_A = cyc_loss*10 + self.disc_loss_B
        self.g_loss_B = cyc_loss*10 + self.disc_loss_A

        self.d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        self.d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()
        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", self.g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", self.g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", self.d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", self.d_loss_B)

    def save_training_images(self, sess, epoch,writer):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                                                                        feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            imsave("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            imsave("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))


    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


    def train(self):


        ''' Training Function '''


        # Load Dataset from the dataset folder
        self.input_setup()  

        #Build the network
        self.model_setup()

        #Loss function calculations
        self.loss_calc()
      
        # Initializing the global variables
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(init_l)

            #Read input to nd array
            self.input_read(sess)


            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter("./output/",sess.graph)

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            for epoch in range(sess.run(self.global_step),200):
                print ("In the epoch ", epoch)
                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                if(save_training_images):
                    self.save_training_images(sess, epoch,writer)

                for idx in range(0,max_images):

                    # Optimizing the G_A network

                    _, fake_B_temp, summary_str,g_loss_A = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ, self.g_loss_A],feed_dict={
                        self.input_A:self.A_input[idx], self.input_B:self.B_input[idx], self.lr:curr_lr})
                    
                    writer.add_summary(summary_str, epoch*max_images + idx)

                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str,d_loss_B = sess.run([self.d_B_trainer, self.d_B_loss_summ,self.d_loss_B],feed_dict={
                        self.input_A:self.A_input[idx], self.input_B:self.B_input[idx], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})

                    writer.add_summary(summary_str, epoch*max_images + idx)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str,g_loss_B = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ,self.g_loss_B],feed_dict={
                        self.input_A:self.A_input[idx], self.input_B:self.B_input[idx], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + idx)
                    
                    
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str ,d_loss_A= sess.run([self.d_A_trainer, self.d_A_loss_summ,self.d_loss_A],feed_dict={
                        self.input_A:self.A_input[idx], self.input_B:self.B_input[idx], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*max_images + idx)
                    
                    self.num_fake_inputs+=1

                    print("Epoch: [%2d] [%6d/%6d]  g_loss_A: %.8f, g_loss_B: %.8f, d_loss_A: %.8f, d_loss_B: %.8f" \
                      % (epoch, idx, max_images, g_loss_A, g_loss_B,d_loss_A, d_loss_B))
            
                sess.run(tf.assign(self.global_step, epoch + 1))


    def test(self):


        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            sess.run(init_l) 

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputA_"+str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputB_"+str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))


def main():
    
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()