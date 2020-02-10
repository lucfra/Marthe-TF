import marthe as mt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

from marthe import build_recursive_model, loss_and_acc, Placeholders
from marthe.utils import *


class MnistExampleConfig(Config):

    def __init__(self, **kwargs):
        self.lr0 = 1.e-2    # smaller that what it should!
        self.mt = 'ffnn'    # model type
        self.bs = 200       # mini batch size
        self.mu = 0.999     # damping factor of mathe
        self.beta = 1.e-5   # hyper learning rate
        self.epo = 10       # approximate total epochs
        super().__init__(**kwargs)


def mnist_example(config: MnistExampleConfig):
    print(config)

    mnist = input_data.read_data_sets('mnist data', one_hot=True)
    train_p = Placeholders()
    valid_p = Placeholders(name='valid')

    def fd(what, where, bs=config.bs):
        batch = what.next_batch(bs) if bs else [what.images, what.labels]
        return {where.x: batch[0], where.y: batch[1]}

    if config.mt == 'ffnn':
        layers = [tf.layers.Dense(800, activation=tf.nn.relu),
                  tf.layers.Dense(10)]
    elif config.mt == 'cnn':
        layers = [lambda x: tf.reshape(x, [-1, 28, 28, 1]),
                  tf.layers.Conv2D(32, 5, padding='same', activation=None),
                  tf.layers.MaxPooling2D(2, 2, 'same'),
                  tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
                  tf.layers.MaxPooling2D(2, 2, 'same'),
                  tf.layers.flatten,
                  tf.layers.Dense(1024, activation=tf.nn.relu),
                  tf.layers.Dense(10)]
    else: raise NotImplementedError()

    train_y = build_recursive_model(train_p.x, layers)  # use different placeholders for training and validation
    valid_y = build_recursive_model(valid_p.x, layers)

    lt, at = loss_and_acc(train_y, train_p.y)
    lv, av = loss_and_acc(valid_y, valid_p.y)

    lr = mt.get_positive_hyperparameter('lr', 0.)

    optim_dict = mt.MomentumOptimizer(lr, 0.9).minimize(lt)

    marthe = mt.Marthe(tf.train.GradientDescentOptimizer(config.beta), mu=config.mu)
    # these are smaller that `usual` since we start with lr=0
    marthe.compute_gradients(lv, optim_dict)

    ss = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    lrr, bv, bt, bi = [], 0., 0., 0
    for i in range(config.epo*55000//config.bs):  # around 25 epochs
        feed_dict = mt.merge_dicts(fd(mnist.train, train_p), fd(mnist.validation, valid_p, bs=config.bs))
        marthe.run(feed_dict)
        lrr.append(lr.eval())
        if i % 100 == 0:
            accuracy_val = av.eval(fd(mnist.validation, valid_p, bs=0))
            if accuracy_val > bv:
                bv = accuracy_val
                bt = av.eval(fd(mnist.test, valid_p, bs=0))
                bi = i
            print('lr:', lr.eval(), 'valid acc:', av.eval(fd(mnist.validation, valid_p, bs=0)))
    plt.plot(lrr)
    plt.show()
    print('best test', bt, 'at iteration', bi)

    return bv


if __name__ == '__main__':
    grid = False
    # example of a grid search
    if grid:
        configs = MnistExampleConfig.grid(mu=[0., 0.9, 0.999, 1.], beta=[1.e-4, 1.e-5, 1.e-6, 1.e-7],
                                          epo=1)  # just 1 empoch for make it quick!
    else:
        # random search, 2 minutes
        configs = MnistExampleConfig.random(
            '00:30', 1,  # 2 min
            beta=lambda rnd: np.exp(np.log(rnd.uniform(1.e-7, 1.e-2))),
            mu=lambda rnd: rnd.choice([0., 0.9, 0.99, 0.999, 0.9999, 1.]),
            epo=1
        )

    results = run_method_on_configs(mnist_example, configs)
    for k, v in results.items():  # returns validation accuracy
        print(k)
        print('result', v)
        print('='*30)

    best = list(sorted(results.items(), key=lambda entry: -entry[1]))[0]

    print('best!:', best[0])
    print(best[1])

    # configs = MnistExampleConfig.random()
