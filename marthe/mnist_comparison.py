from collections import Iterator
from collections import defaultdict
from datetime import timedelta

import far_ho.examples.datasets as dts

from marthe import *
from marthe import AllPlaceholders

from tensorflow.examples.tutorials.mnist import input_data


def linear_model(out_size=10):
    return [tf.layers.Dense(units=out_size, activation=None)]


def ffnn(out_size=10):
    return [tf.layers.Dense(800, activation=tf.nn.relu),
            tf.layers.Dense(out_size)]


def cnn(out_size=10):
    return [lambda x: tf.reshape(x, [-1, 28, 28, 1]),
            tf.layers.Conv2D(32, 5, padding='same', activation=None),
            tf.layers.MaxPooling2D(2, 2, 'same'),
            tf.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
            tf.layers.MaxPooling2D(2, 2, 'same'),
            tf.layers.flatten,
            tf.layers.Dense(1024, activation=tf.nn.relu),
            tf.layers.Dense(out_size)]


def load_mnist():
    mnist = input_data.read_data_sets('mnist data', one_hot=True)
    return dts.Datasets.from_list([
        dts.Dataset(mnist.train.images, mnist.train.labels),
        dts.Dataset(mnist.validation.images, mnist.validation.labels),
        dts.Dataset(mnist.test.images, mnist.test.labels),
    ])


class MnistExpConfig(Config):

    def __init__(self, **kwargs):
        self.lr0 = 0.01
        self.mf = 0.9  # momentum
        self.bs = 128  # mini-batch size
        self.epo = 10  # epochs
        self.cke = 0.05  # check every iters
        self.pat = 200  # patience (times cke)
        self.seed = 1
        self.mod = 'cnn'
        self._opt = None
        super().__init__(**kwargs)

    def model(self):
        if self.mod == 'linear': return linear_model(10)
        elif self.mod == 'ffnn': return ffnn(10)
        elif self.mod == 'cnn': return cnn(10)
        else: raise NotImplementedError('{} not known'.format(self.mod))

    def lr_and_step(self, loss, gs, iters_per_epoch):
        step = tf.train.MomentumOptimizer(self.lr0, self.mf).minimize(loss)
        return tf.convert_to_tensor(self.lr0), lambda fd: step.run(fd)


class MnistExpConfigExpDecay(MnistExpConfig):

    def __init__(self, **kwargs):
        self.dr = 1.
        super().__init__(**kwargs)

    def lr_and_step(self, loss, gs, iters_per_epoch):
        lr = tf.train.exponential_decay(self.lr0, gs, iters_per_epoch, self.dr)
        step = tf.train.MomentumOptimizer(lr, self.mf).minimize(loss, global_step=gs)
        return lr, lambda fd: step.run(fd)


class _ExpConfWithValidMnist(MnistExpConfig):
    pass


class MnistExpConfigMarthe(_ExpConfWithValidMnist):

    def __init__(self, **kwargs):
        self.beta = 1.e-6
        self.mu = 0.9
        super().__init__(**kwargs)

    def _lr_and_step(self, loss, mu):
        lr = get_positive_hyperparameter('lr', self.lr0)

        optimizer = MomentumOptimizer(lr, self.mf)

        opt_dict = optimizer.minimize(loss[0])
        marthe = Marthe(tf.train.GradientDescentOptimizer(self.beta), mu=mu)
        marthe.compute_gradients(loss[1], opt_dict)
        self._opt = marthe
        return lr, marthe.run

    def lr_and_step(self, loss, gs, iters_per_epoch):
        return self._lr_and_step(loss, self.mu)


class MnistExpConfigHD(MnistExpConfigMarthe):
    pass


def mnist_exp(mnist, config: MnistExpConfig):
    if isinstance(config, Iterator) or isinstance(config, list):
        return [mnist_exp(mnist, c) for c in config]
    ss = setup_tf(config.seed)

    print(config)

    statistics = defaultdict(list)  # TODO add decent version of config in statistics (no _opt!)
    # statistics[config] = str(config)

    plcs = AllPlaceholders(mnist)
    model = config.model()
    outs = plcs.build_recursive_outputs(model)
    lsac = plcs.losses_and_accs(outs)

    print(tf.trainable_variables())

    iter_per_epoch = mnist.train.num_examples//config.bs

    lr, step = config.lr_and_step(
        [lsac.train.loss, lsac.val.loss] if isinstance(config, _ExpConfWithValidMnist) else lsac.train.loss,
        tf.Variable(0, False, name='global_step'), iter_per_epoch)

    suppliers = plcs.create_suppliers([config.bs, config.bs, None])
    tf.global_variables_initializer().run()

    es = early_stopping(
        config.pat, config.epo*iter_per_epoch, on_accept=lambda accept_iters, accept_val_acc:
        update_append(statistics, es_accept=(accept_iters, accept_val_acc, lsac.test.acc.eval(suppliers.test())))
    )
    full_val = mnist.validation.create_supplier(plcs.plcs.val.x, plcs.plcs.val.y, batch_size=None)
    train_2 = mnist.train.create_supplier(plcs.plcs.train.x, plcs.plcs.train.y, batch_size=10000)

    start_time = time.time()
    for i in es:
        if isinstance(config, MnistExpConfigHD):
            ti = suppliers.train(max(i, i - 1))  # use ''validation'' placeholder for train
            fd = merge_dicts(suppliers.train(i),
                             {plcs.plcs.val.x: ti[plcs.plcs.train.x],
                              plcs.plcs.val.y: ti[plcs.plcs.train.y]})
        elif isinstance(config, _ExpConfWithValidMnist):
            fd = merge_dicts(suppliers.train(i), suppliers.val(i))
        else:
            fd = suppliers.train(i)
        step(fd)  # for the baseline the validation supplier doesn't matter!

        learning_rate = ss.run(lr)
        update_append(statistics, learning_rate=learning_rate)
        if i % int(config.cke*iter_per_epoch) == 0:
            # compute full validation accuracy

            train_acc = np.mean([
                ss.run(lsac.train.acc, train_2(i))
                for i in range(mnist.train.num_examples // 10000)
            ])

            validation_accuracy = lsac.val.acc.eval(full_val())

            test_accuracy = lsac.test.acc.eval(suppliers.test())
            update_append(statistics,
                          ind=i,
                          hg_clip_cout=config._opt.hg_clip_counter.eval(),
                          train_accuracy=train_acc,
                          validation_accuracy=validation_accuracy,
                          test_accuracy=test_accuracy, elapsed_time=str(timedelta(seconds=time.time() - start_time)),
                          elapsed_time_sec=time.time() - start_time)

            print(i, '\t', train_acc, '\t', validation_accuracy, '\t', test_accuracy, '\t', learning_rate,
                  '\t', config._opt.hg_clip_counter.eval())
            try:
                es.send(validation_accuracy)
            except StopIteration:
                pass
            gz_write(statistics, config.str_for_filename())

    # end
    end_string = '{} \t {} .'.format(time.asctime(), config.str_for_filename()) + \
                 '\n iters, valid, test = {}  \t| total time {} \n'.format(
                     statistics['es_accept'][-1], statistics['elapsed_time'][-1]
                 )
    with open('MNIST_ledger.txt', 'a+') as f:
        f.writelines(end_string)

    return statistics['es_accept'][-1]


if __name__ == '__main__':
    # TODO do the script.
    configs = MnistExpConfigMarthe.grid(lr0=0.05, mu=[0., 0.5, 0.9, 0.999, 1.],
                                        beta=[1.e-4, 1.e-5, 1.e-6], epo=15)
    # configs = MnistExpConfigMarthe.grid(lr0=0.0, mu=[0., 0.5, 0.9, 0.999, 1.],
    #                                     beta=[1.e-4, 1.e-5, 1.e-6],
    #                                     epo=20, mod='ffnn')

    data = load_mnist()
    mnist_exp(data, configs)
