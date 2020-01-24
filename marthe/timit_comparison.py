from collections import Iterator
from collections import defaultdict
from datetime import timedelta

import far_ho.examples.datasets as dts

from marthe import *
from marthe import AllPlaceholders

from tensorflow.examples.tutorials.mnist import input_data

import marthe.timit_main as tmm


class TimitCompExpConfig(Config):

    def __init__(self, **kwargs):
        self.lr0 = 0.05
        self.mf = 0.5  # momentum
        self.bs = 200  # mini-batch size
        self.epo = 15  # epochs
        self.cke = 0.1  # check every iters
        self.pat = 100  # large patience
        self.seed = 1
        self._opt = None
        # self.mod = 'cnn'
        super().__init__(**kwargs)

    def model(self):
        return tmm.timit_ffnn()

    def lr_and_step(self, loss, gs, iters_per_epoch):
        step = tf.train.MomentumOptimizer(self.lr0, self.mf).minimize(loss)
        return tf.convert_to_tensor(self.lr0), lambda fd: step.run(fd)


class TimitCompExpConfigDecay(TimitCompExpConfig):

    def __init__(self, **kwargs):
        self.dr = 1.
        super().__init__(**kwargs)

    def lr_and_step(self, loss, gs, iters_per_epoch):
        lr = tf.train.exponential_decay(self.lr0, gs, iters_per_epoch, self.dr)
        step = tf.train.MomentumOptimizer(lr, self.mf).minimize(loss, global_step=gs)
        return lr, lambda fd: step.run(fd)


class _ExpConfWithValidTimitComp(TimitCompExpConfig):
    pass


class TimitCompExpConfigMarthe(_ExpConfWithValidTimitComp):

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


class TimitCompExpConfigHD(TimitCompExpConfigMarthe):
    pass


def timit_comp_exp(timit, config: TimitCompExpConfig):
    if isinstance(config, Iterator) or isinstance(config, list):
        return [timit_comp_exp(timit, c) for c in config]
    ss = setup_tf(config.seed)

    print(config)

    statistics = defaultdict(list)

    plcs = AllPlaceholders(timit)
    model = config.model()
    outs = plcs.build_recursive_outputs(model)
    lsac = plcs.losses_and_accs(outs)

    print(tf.trainable_variables())

    iter_per_epoch = timit.train.num_examples//config.bs

    lr, step = config.lr_and_step(
        [lsac.train.loss, lsac.val.loss] if isinstance(config, _ExpConfWithValidTimitComp) else lsac.train.loss,
        tf.Variable(0, False, name='global_step'), iter_per_epoch)

    suppliers = plcs.create_suppliers([config.bs, config.bs, None])
    tf.global_variables_initializer().run()

    es = early_stopping(
        config.pat, config.epo*iter_per_epoch, on_accept=lambda accept_iters, accept_val_acc:
        update_append(statistics, es_accept=(accept_iters, accept_val_acc, lsac.test.acc.eval(suppliers.test())))
    )
    train_2 = timit.train.create_supplier(plcs.plcs.train.x, plcs.plcs.train.y, batch_size=10000)
    valid_2 = timit.validation.create_supplier(plcs.plcs.val.x, plcs.plcs.val.y, batch_size=10000)

    start_time = time.time()
    for i in es:
        if isinstance(config, TimitCompExpConfigHD):
            ti = suppliers.train(max(i, i - 1))  # use ''validation'' placeholder for train
            fd = merge_dicts(suppliers.train(i),
                             {plcs.plcs.val.x: ti[plcs.plcs.train.x],
                              plcs.plcs.val.y: ti[plcs.plcs.train.y]})
        elif isinstance(config, _ExpConfWithValidTimitComp):
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
                for i in range(10)  # all the training set is too big... just check first chunk of data...
            ])

            validation_accuracy = np.mean([
                ss.run(lsac.val.acc, valid_2(i))
                for i in range(timit.validation.num_examples//10000)
            ])

            test_accuracy = lsac.test.acc.eval(suppliers.test())
            update_append(statistics,
                          ind=i,
                          clipped_hg=config._opt.hg_clip_counter.eval(),
                          train_accuracy=train_acc,
                          validation_accuracy=validation_accuracy,
                          test_accuracy=test_accuracy, elapsed_time=str(timedelta(seconds=time.time() - start_time)),
                          elapsed_time_sec=time.time() - start_time)

            print(i, '\t', train_acc, '\t', validation_accuracy, '\t', test_accuracy, '\t', learning_rate, '\t',
                  config._opt.hg_clip_counter.eval())
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
    configs = TimitCompExpConfigMarthe.grid(lr0=0.01, mu=[0., 0.5, 0.9, 0.999, 1.],
                                        beta=[1.e-4, 1.e-5, 1.e-6])
    data = tmm.load_timit(only_primary=True, context=5)
    timit_comp_exp(data, configs)
