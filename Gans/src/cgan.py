import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np


class GANs(object):
    """
    Vanilla GAN implementation
    """

    def __init__(self, lr, layers, z_dim, y_dim, batch_size, num_steps, print_loss_at=1000, verbose=True,
                 save_samples=True):
        """
        Intialize all the class variables with the argument variables
        :param lr: learning rate
        :param layers: hidden layer size
        :param z_dim: size of the noise
        :param z_dim: size of the condition
        :param batch_size: training batch size
        :param num_steps: number of iteration
        :param print_loss_at: print loss on multiple of this
        :param verbose: print loss
        :param save_samples: save samples
        """

        self.mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        self.lr = lr
        self.layers = layers
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.print_loss_at = print_loss_at
        self.verbose = verbose
        self.disc_var = None
        self.gen_var = None
        self.saver = None
        self.save_samples = save_samples
        self.gen_image = None
        if self.save_samples:
            if not os.path.exists("outputs/output_cgan/"):
                os.mkdir("outputs/output_cgan/")
        if not os.path.exists("results/result_cgan/"):
            os.mkdir("results/result_cgan/")

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

    def get_placeholders(self):
        """

        :return:
        """
        x = tf.placeholder(name="input_image", shape=[None, 784], dtype=tf.float32)
        y = tf.placeholder(name="condition", shape=[None, self.y_dim], dtype=tf.float32)
        z = tf.placeholder(name="noise", shape=[None, self.z_dim], dtype=tf.float32)

        return x, y, z

    def get_noise(self, size):
        """

        :return:
        """
        return np.random.uniform(-1., 1., size=[size, self.z_dim])

    def save_sample(self, gen_images, iteration=None):
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
            plt.savefig('outputs/output_cgan/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
        else:
            plt.savefig('results/result_cgan/result.png', bbox_inches='tight')
        plt.close()

    def generator(self, y, z):
        """

        :param z:
        :return:
        """
        with tf.variable_scope("generator", reuse=False):
            z_dash = tf.concat(values=[z, y], axis=1, name="gen_concat")
            g_w1 = self.get_weights([self.z_dim + self.y_dim, self.layers], "g_w1")
            g_b1 = self.get_biases([self.layers], "g_b1")
            g_w2 = self.get_weights([self.layers, 784], "g_w2")
            g_b2 = self.get_biases([784], "g_b2")

            h1 = tf.nn.relu(tf.add(tf.matmul(z_dash, g_w1), g_b1), name="gen_h1")
            out_logit = tf.add(tf.matmul(h1, g_w2), g_b2, name="gen_out_logit")
            gen_prob = tf.nn.sigmoid(out_logit, name="gen_prob")
            self.gen_var = [g_w2, g_b2, g_w1, g_b1]

        return gen_prob

    def discriminator(self, x, y, reuse):
        """

        :param x:
        :param reuse:
        :return:
        """

        with tf.variable_scope("discriminator", reuse=reuse):
            x_dash = tf.concat(values=[x, y], axis=1, name="disc_concat")
            d_w1 = self.get_weights([784 + self.y_dim, self.layers], "d_w1")
            d_b1 = self.get_biases([self.layers], "d_b1")
            d_w2 = self.get_weights([self.layers, 1], "d_w2")
            d_b2 = self.get_biases([1], "d_b2")

            h1 = tf.nn.relu(tf.add(tf.matmul(x_dash, d_w1), d_b1), name="dis_h1")
            out_logit = tf.add(tf.matmul(h1, d_w2), d_b2, name="dis_out_logit")
            dis_prob = tf.nn.sigmoid(out_logit, name="dis_prob")

            self.disc_var = [d_w2, d_b2, d_w1, d_b1]

        return out_logit, dis_prob

    def get_loss(self, x, y, z):
        """

        :param x:
        :param z:
        :return:
        """
        self.gen_image = self.generator(y, z)
        disc_logit_real, disc_prob_real = self.discriminator(x, y, False)
        disc_logit_fake, disc_prob_fake = self.discriminator(self.gen_image, y, True)

        disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_real,
                                                                                labels=tf.ones_like(disc_logit_real)))

        disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_fake,
                                                                                labels=tf.zeros_like(disc_logit_fake)))

        disc_loss = disc_fake_loss + disc_real_loss

        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logit_fake,
                                                                          labels=tf.ones_like(disc_logit_fake)))

        return disc_loss, gen_loss

    def train(self):
        """

        :return:
        """

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            x, y, z = self.get_placeholders()
            print(x,y,z)
            disc_loss, gen_loss = self.get_loss(x, y, z)
            disc_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(disc_loss, var_list=self.disc_var)
            gen_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(gen_loss, var_list=self.gen_var)
            self.saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_steps):

                x_image, y_image = self.mnist.train.next_batch(self.batch_size)
                z_noise = self.get_noise(self.batch_size)

                _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: x_image, z: z_noise, y:y_image})
                _, g_loss = sess.run([gen_opt, gen_loss], feed_dict={z: z_noise, y: y_image})

                if self.verbose and i % self.print_loss_at == 0:
                    msg = "Iteration: {}\nDiscriminator Loss: {}\nGenerator_loss: {}\n\n".format(i, d_loss, g_loss)
                    print(msg)
                    self.saver.save(sess, "models/model_cgan/")

                if self.save_samples and i % self.print_loss_at == 0:
                    z_noise = self.get_noise(16)
                    print(y_image[0, :].shape)
                    print(y_image[0].shape)
                    gen_images = sess.run([self.gen_image], feed_dict={z: z_noise, y: y_image[0:16, :]})[0]
                    self.save_sample(gen_images, i)

    def generate(self, condition):
        l = []
        temp = np.eye(10)
        for i in condition:
            cond = temp[i]
            l.append(cond)
        l = np.array(l)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            x, y, z = self.get_placeholders()
            disc_loss, gen_loss = self.get_loss(x, y, z)
            self.saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, "./models/model_cgan/")
            z_noise = self.get_noise(16)
            gen_images = sess.run([self.gen_image], feed_dict={z: z_noise, y: l})[0]
            self.save_sample(gen_images)


gan = GANs(0.0003, 256, 100, 10, 128, 100000)
# gan.train()
gan.generate([1, 3, 2, 5, 6, 7, 9, 0, 3, 8, 8, 1, 1, 3, 9, 7])
