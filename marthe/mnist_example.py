import marthe as mt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

from marthe import build_recursive_model, loss_and_acc, Placeholders


def mnist_example(ffnn=False):
    # ffnn: use ffnn or CNN
    mnist = input_data.read_data_sets('mnist data', one_hot=True)
    train_p = Placeholders()
    valid_p = Placeholders(name='valid')

    def fd(what, where, bs=128):
        batch = what.next_batch(bs) if bs else [what.images, what.labels]
        return {where.x: batch[0], where.y: batch[1]}

    if ffnn:
        layers = [tf.layers.Dense(800, activation=tf.nn.relu),
                  tf.layers.Dense(10)]
    else:
        layers = [lambda x: tf.reshape(x, [-1, 28, 28, 1]),
                  tf.layers.Conv2D(32, 5, padding='same', activation=None),
                  tf.layers.MaxPooling2D(2, 2, 'same'),
                  tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
                  tf.layers.MaxPooling2D(2, 2, 'same'),
                  tf.layers.flatten,
                  tf.layers.Dense(1024, activation=tf.nn.relu),
                  tf.layers.Dense(10)]

    train_y = build_recursive_model(train_p.x, layers)  # use different placeholders for training and validation
    valid_y = build_recursive_model(valid_p.x, layers)

    lt, at = loss_and_acc(train_y, train_p.y)
    lv, av = loss_and_acc(valid_y, valid_p.y)

    lr = mt.get_positive_hyperparameter('lr', 0.)

    optim_dict = mt.MomentumOptimizer(lr, 0.9).minimize(lt)

    marthe = mt.Marthe(beta=1.e-8 if ffnn else 1.e-10)  # these are smaller that `usual` since we start with lr=0
    marthe.compute_gradients(lv, optim_dict)

    ss = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    lrr, bv, bt, bi = [], 0., 0., 0
    for i in range(12000):  # around 25 epochs
        feed_dict = mt.merge_dicts(fd(mnist.train, train_p), fd(mnist.validation, valid_p, bs=200))
        marthe.run(feed_dict)
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
