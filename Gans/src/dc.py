import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


class BatchNorm(object):

    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, layer, training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            bn = tf.contrib.layers.batch_norm(layer, decay=self.momentum, epsilon=self.epsilon,
                                              scale=True, is_training=training, scope=self.name)
        return bn


class Ops(object):

    def __init__(self):
        self.padding = 'SAME'
        self.k_h = 5
        self.k_w = 5
        self.s_h = 2
        self.s_w = 2
        self.leak = 0.2

    @staticmethod
    def get_weights(shape, name):
        """

        :param shape:
        :param name:
        :return:
        """
        return tf.get_variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(), name=name)

    @staticmethod
    def get_biases(shape, name):
        """

        :param shape:
        :param name:
        :return:
        """
        return tf.get_variable(initializer=tf.zeros(shape=shape), name=name)

    def conv2d(self, x_image, out_channel, name, reuse):

        in_channel = x_image.get_shape().as_list()[-1]

        with tf.variable_scope(name, reuse=reuse):

            c_w = self.get_weights(shape=[self.k_h, self.k_w, in_channel, out_channel], name=name+'_w')
            b_w = self.get_biases(shape=[out_channel], name=name+'_b')

            conv_layer = tf.nn.conv2d(x_image, c_w, strides=[1, self.s_h, self.s_w, 1], padding=self.padding)
            conv_layer += b_w

            return conv_layer

    def deconv(self, x_image, out_shape, name):

        in_channel = x_image.get_shape().as_list()[-1]
        out_channel = out_shape[-1]

        # print("8888888888888888888")
        with tf.variable_scope(name):

            de_w = self.get_weights(shape=[self.k_h, self.k_w, out_channel, in_channel], name=name+"_w")
            # print("input weight", de_w)
            # print("out_shape", out_shape)
            # print("x_image.get_shape().as_list()", x_image.get_shape().as_list())
            de_b = self.get_biases(shape=[out_channel], name=name+"_b")

            deconv = tf.nn.conv2d_transpose(x_image, de_w, out_shape, strides=[1, self.s_h, self.s_w, 1],
                                            padding=self.padding)
            deconv = tf.add(deconv, de_b)

        return deconv

    def lrelu(self, x, name):

        return tf.maximum(x, x*self.leak, name=name)

    def linear(self, x, out_size, name, reuse=False):

        in_shape = x.get_shape().as_list()[-1]
        # print("name", name)
        # print(in_shape)
        # print(x.get_shape())
        # print(out_size)
        with tf.variable_scope(name, reuse=reuse):

            l_w = self.get_weights(shape=[in_shape, out_size], name=name+"_w")
            l_b = self.get_biases(shape=[out_size], name=name+"_b")

            return tf.add(tf.matmul(x, l_w), l_b)


class DCGAN(object):

    def __init__(self, batch_size=128, num_steps=100, z_dim=100, gf_dim=64, df_dim=64,
                 input_channel=1):

        self.ops = Ops()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.in_channel = input_channel
        self.num_steps = num_steps
        self.d_bn1 = BatchNorm(name="disc_bn1")
        self.d_bn2 = BatchNorm(name="disc_bn2")
        self.d_bn3 = BatchNorm(name='disc_bn3')
        self.g_bn1 = BatchNorm(name="gen_bn1")
        self.g_bn2 = BatchNorm(name="gen_bn2")
        self.g_bn3 = BatchNorm(name="gen_bn3")
        self.g_bn4 = BatchNorm(name='gen_bn4')
        self.mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        self.input = None
        self.z = None
        self.d_real = None
        self.g = None
        self.d_fake = None
        self.saver = None
        self.d_loss = None
        self.g_loss = None
        self.d_opt = None
        self.g_opt = None
        self.graph = None
        self.reuse = False
        self.logit_real = None
        self.logit_fake = None
        self.g_var_list = []
        self.d_var_list = []
        self.build()

    def create_placeholders(self):

        self.input = tf.placeholder(tf.float32, [self.batch_size, 28, 28, self.in_channel], name='input_image')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="z")

    def build(self):

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_placeholders()
            self.g = self.generator(self.z)
            self.d_real, self.logit_real = self.discriminator(self.input, reuse=False)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.d_fake, self.logit_fake = self.discriminator(self.g)
            self.loss()
            t_vars = tf.trainable_variables()
            self.d_var_list = [var for var in t_vars if 'disc_' in var.name]
            self.g_var_list = [var for var in t_vars if 'gen_' in var.name]
            self.optimizer()
            self.saver = tf.train.Saver()

    @staticmethod
    def save_sample(gen_images, iteration=None):
        """

        :param gen_images:
        :param iteration:
        :return:
        """

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, image in enumerate(gen_images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(image.reshape(28, 28), cmap='Greys_r')
        if iteration is not None:
            plt.savefig('outputs/output_dcgan/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
        else:
            plt.savefig('results/result_dcgan/result.png', bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_noise(size):

        return np.random.uniform(-1, 1, [size, 100]).astype(np.float32)

    @staticmethod
    def sigmoid_cross_entropy(x, y):

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)

    def optimizer(self):

        self.g_opt = tf.train.AdamOptimizer(learning_rate=0.003).minimize(self.g_loss, var_list=self.g_var_list)
        self.d_opt = tf.train.AdamOptimizer(learning_rate=0.003).minimize(self.d_loss, var_list=self.g_var_list)

    def loss(self):

        d_loss_real = tf.reduce_mean(self.sigmoid_cross_entropy(x=self.d_real, y=tf.ones_like(self.logit_real)))
        d_loss_fake = tf.reduce_mean(self.sigmoid_cross_entropy(x=self.d_fake, y=tf.zeros_like(self.logit_fake)))

        self.d_loss = d_loss_fake + d_loss_real

        self.g_loss = tf.reduce_mean(self.sigmoid_cross_entropy(x=self.d_fake, y=tf.ones_like(self.logit_fake)))

    def generator(self, z):

        with tf.variable_scope("generator", reuse=False):
            s_h, s_w = 28, 28
            s_h2, s_w2 = int(math.ceil(float(s_h) / float(2))), int(math.ceil(float(s_w) / float(2)))
            s_h4, s_w4 = int(math.ceil(float(s_h2) / float(2))), int(math.ceil(float(s_w2) / float(2)))
            s_h8, s_w8 = int(math.ceil(float(s_h4) / float(2))), int(math.ceil(float(s_w4) / float(2)))
            s_h16, s_w16 = int(math.ceil(float(s_h8) / float(2))), int(math.ceil(float(s_w8) / float(2)))

            # print(s_h, s_h2, s_h4, s_h8, s_h16)
            # print("====================================================================")
            # print("====================================================================")
            # print("====================================================================")
            print(z)
            h1 = self.ops.linear(z, self.gf_dim*8*s_h16*s_w16, name="gen_h1")
            print(h1)
            h1 = tf.reshape(h1, shape=[-1, s_h16, s_w16, self.gf_dim*8])
            print(h1)
            h1 = tf.nn.relu(self.g_bn1(h1))
            # print(h1)

            shape_h2 = [self.batch_size, s_h8, s_w8, self.gf_dim*4]
            # print(shape_h2)
            h2 = self.ops.deconv(h1, shape_h2, name="gen_deconv1")
            h2 = tf.nn.relu(self.g_bn2(h2))

            shape_h3 = [self.batch_size, s_h4, s_w4, self.gf_dim * 2]
            h3 = self.ops.deconv(h2, shape_h3, name="gen_deconv2")
            h3 = tf.nn.relu(self.g_bn3(h3))

            shape_h4 = [self.batch_size, s_h2, s_w2, self.gf_dim]
            h4 = self.ops.deconv(h3, shape_h4, name="gen_deconv3")
            h4 = tf.nn.relu(self.g_bn4(h4))

            shape_h5 = [self.batch_size, s_h, s_w, self.in_channel]
            h5 = self.ops.deconv(h4, shape_h5, name="gen_deconv4")
            # if not self.reuse:
            #     self.reuse = True
            # print("6767676767", h5)
            return tf.nn.tanh(h5, name="gen_tanh")

    def discriminator(self, x, reuse=True):
        # print("212123", x)
        with tf.variable_scope("discriminator", reuse):
            conv1 = self.ops.lrelu(self.ops.conv2d(x, self.df_dim, reuse=reuse, name="disc_conv1"), name="disc_conv1_lrelu")
            conv2 = self.ops.lrelu(self.d_bn1(self.ops.conv2d(conv1, self.df_dim*2, reuse=reuse,  name="disc_conv2"), reuse=reuse), name="disc_conv2_lrelu")
            conv3 = self.ops.lrelu(self.d_bn2(self.ops.conv2d(conv2, self.df_dim*4, reuse=reuse,  name="disc_conv3"), reuse=reuse), name="disc_conv3_lrelu")
            conv4 = self.ops.lrelu(self.d_bn3(self.ops.conv2d(conv3, self.df_dim*8, reuse=reuse,  name="disc_conv4"), reuse=reuse), name="disc_conv4_lrelu")
            # print(conv4)
            flat = tf.reshape(conv4, shape=[self.batch_size, -1])
            # print(";;;;;;;;;;;;;;;;;;", flat)
            h1 = self.ops.linear(flat, 1, name="disc_h1", reuse=reuse)

            return tf.nn.sigmoid(h1, name="disc_sigmoid"), h1

    def train(self):

        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())
            for i in range(self.num_steps):
                x_image, _ = self.mnist.train.next_batch(self.batch_size)
                z = self.get_noise(self.batch_size)
                print(z.shape)
                # x_image = np.reshape(x_image, newshape=(self.batch_size, 28, 28, 1))
                x_image = np.reshape(x_image, newshape=(self.batch_size, 28, 28, 1))
                _, d_loss = sess.run([self.d_opt, self.d_loss], feed_dict={self.input: x_image, self.z: z})

                _, g_loss = sess.run([self.g_opt, self.g_loss], feed_dict={self.z: z})
                if i % 10 == 0:
                    print("\ngen_loss {}\ndisc_loss {}".format(g_loss, d_loss))
                    gen_images = self.get_noise(16)
                    gen_images = sess.run([self.g], feed_dict={self.z: gen_images})
                    self.save_sample(gen_images, i)
                    self.saver.save(sess, "models/model_dcgan/")


dcgan = DCGAN()
dcgan.train()
