import marthe as mt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import reduce

import matplotlib.pyplot as plt

class Placeholders:

    def __init__(self, dim_x=28*28, dim_y=10, name='train'):
        with tf.name_scope(name):
            dim_x = mt.as_list(dim_x)
            self.x = tf.placeholder(tf.float32, shape=[None] + dim_x, name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, dim_y], name='y')


def rec_model(x, layers):
    return reduce(lambda t, nl: nl(t), layers, x)


def loss_and_acc(out, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                              name='accuracy')
    return loss, accuracy


def mnist_example():

    # data
    mnist = input_data.read_data_sets('mnist data', one_hot=True)
    train_p = Placeholders()
    valid_p = Placeholders(name='valid')

    def fd(what, where, bs=256):
        batch = what.next_batch(bs) if bs else [what.images, what.labels]
        return {where.x: batch[0], where.y: batch[1]}

    layers = [tf.layers.Dense(800, activation=tf.nn.relu),
              tf.layers.Dense(10)]

    layers = [lambda x: tf.reshape(x, [-1, 28, 28, 1]),
              tf.layers.Conv2D(32, 5, padding='same', activation=tf.nn.relu),
              tf.layers.MaxPooling2D(2, 2, 'same'),
              tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
              tf.layers.MaxPooling2D(2, 2, 'same'),
              tf.layers.flatten,
              tf.layers.Dense(1024, activation=tf.nn.relu),
              tf.layers.Dense(10)]

    ty = rec_model(train_p.x, layers)
    vy = rec_model(valid_p.x, layers)

    lt, at = loss_and_acc(ty, train_p.y)
    lv, av = loss_and_acc(vy, valid_p.y)

    lr = mt.get_hyper_box_constraints('lr', 0., 0., 1.)

    optim_dict = mt.MomentumOptimizer(lr, 0.9).minimize(lt)

    beta_prime = tf.placeholder(tf.float32)
    marthe = mt.Marthe(tf.train.GradientDescentOptimizer(beta_prime), alpha=1.e-8)
    marthe.compute_gradients(lv, optim_dict)

    ss = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    lrr, bv, bt, bi = [], 0., 0., 0
    for i in range(12000):
        restarted = marthe.run(mt.merge_dicts(fd(mnist.train, train_p), fd(mnist.validation, valid_p)))
        if restarted: lrr, bv, bt, bi = [], 0., 0., 0
        lrr.append(lr.eval())
        if i % 100 == 0:
            accuracy_val = av.eval(fd(mnist.validation, valid_p, bs=0))
            if accuracy_val > bv:
                bv = accuracy_val
                bt = av.eval(fd(mnist.test, valid_p, bs=0))
                bi = i
            print('lr:', lr.eval(), 'beta_prime:', marthe.beta_val, 'mu:', marthe.mu_val,
                  'valid acc:', av.eval(fd(mnist.validation, valid_p, bs=0)))
    plt.plot(lrr)
    plt.show()
    print('best test', bt, 'at iteration', bi)


if __name__ == '__main__':
    mnist_example()