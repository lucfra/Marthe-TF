from datetime import timedelta
from collections import defaultdict
from collections import Iterator

from marthe import *
import tensorflow as tf
import far_ho.examples.datasets as dts

import intervaltree as it

import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

import scipy.sparse as sp

from marthe import AllPlaceholders

UCI_DATASET_NAMES = ['wine', 'breast_cancer', 'digits', 'diabetes', '20news10']


def load(dataset_name, valid_perc=.15, test_perc=.15):
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'wine':
        data = datasets.load_wine()
    elif dataset_name == 'breast_cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'digits':
        data = datasets.load_digits()
    elif dataset_name == 'diabetes':
        data = datasets.load_diabetes()
    elif dataset_name == '20news10':
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(subset='all', categories=categories)
        vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
        X_counts = vectorizer.fit_transform(data.data).toarray()
        transformer = TfidfTransformer(smooth_idf=False)
        features = transformer.fit_transform(X_counts).todense()
    else:
        raise AttributeError('dataset not available')

    if dataset_name != '20news10':
        features = data.data
    y = data.target

    if dataset_name != 'diabetes':
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:  # for binary tasks
            ys = np.hstack([ys, 1 - ys])
    else: ys = y

    # noinspection PyUnboundLocalVariable
    n = features.shape[0]  # number of examples

    from sklearn.model_selection import train_test_split

    tv_x, test_x, tvy, test_y = train_test_split(features, ys, test_size=test_perc)
    train_x, valid_x, train_y, valid_y = train_test_split(tv_x, tvy, test_size=valid_perc)

    return dts.Datasets.from_list([dts.Dataset(train_x, train_y),
                                  dts.Dataset(valid_x, valid_y),
                                   dts.Dataset(test_x, test_y)])


def log_reg(out_size):
    return [tf.layers.Dense(units=out_size, activation=None)]


class UCIExpConfig(Config):

    def __init__(self, **kwargs):
        self.lr0 = 0.01
        self.mu = 0.9  # momentum
        self.bs = 200  # mini-batch size
        self.epo = 20  # epochs
        self.cke = 0.1  # check every iters
        self.pat = 20  # patience (times cke)
        self.seed = 1
        self.dts_name = 'wine'
        super().__init__(**kwargs)

    def lr_and_step(self, loss, gs, iters_per_epoch):
        step = tf.train.MomentumOptimizer(self.lr0, self.mu).minimize(loss)
        return tf.convert_to_tensor(self.lr0), lambda fd: step.run(fd)


class _ExpConfWithValidUCI(UCIExpConfig):
    pass


class UCIExpConfigMarthe(_ExpConfWithValidUCI):

    def __init__(self, **kwargs):
        self.beta = 1.e-6
        super().__init__(**kwargs)

    def lr_and_step(self, loss, gs, iters_per_epoch):
        lr = get_positive_hyperparameter('lr', self.lr0)

        optimizer = MomentumOptimizer(lr, self.mu)

        opt_dict = optimizer.minimize(loss[0])
        marthe = Marthe(beta=self.beta)
        marthe.compute_gradients(loss[1], opt_dict)
        return lr, marthe.run


class UCIExpConfigRTHO(_ExpConfWithValidUCI):

    def __init__(self, **kwargs):
        self.beta = 1.e-6
        super().__init__(**kwargs)

    def _lr_and_step(self, loss, mu):
        lr = get_positive_hyperparameter('lr', self.lr0)

        optimizer = MomentumOptimizer(lr, self.mu)

        opt_dict = optimizer.minimize(loss[0])
        marthe = Marthe(tf.train.GradientDescentOptimizer(self.beta), mu=mu)
        marthe.compute_gradients(loss[1], opt_dict)
        return lr, marthe.run

    def lr_and_step(self, loss, gs, iters_per_epoch):
        return self._lr_and_step(loss, 1.)


class TimitExpConfigHD(UCIExpConfigRTHO):

    def lr_and_step(self, loss, gs, iters_per_epoch):
        return self._lr_and_step(loss, 0.)


class UCIExpConfigExpDecay(UCIExpConfig):

    def __init__(self, **kwargs):
        self.dr = 1.
        super().__init__(**kwargs)

    def lr_and_step(self, loss, gs, iters_per_epoch):
        lr = tf.train.exponential_decay(self.lr0, gs, iters_per_epoch, self.dr)
        step = tf.train.MomentumOptimizer(lr, self.mu).minimize(loss, global_step=gs)
        return lr, lambda fd: step.run(fd)


# def timit_exp(config: UCIExpConfig):
#     if isinstance(config, Iterator) or isinstance(config, list):
#         return [timit_exp(c) for c in config]
#     ss = setup_tf(config.seed)
#
#     print(config)
#     timit = load_timit(only_primary=True, context=5, small=config.small_dts)
#     # find a good way to load the data only once  (use DataConfig!)
#     print(*[d.num_examples for d in timit])
#
#     statistics = defaultdict(list)
#     iters_per_epoch = timit.train.num_examples // config.bs
#     statistics['iters per epoch'] = iters_per_epoch
#
#     plcs = AllPlaceholders(timit)
#     model = timit_ffnn()
#     outs = plcs.build_recursive_outputs(model)
#     lsac = plcs.losses_and_accs(outs)
#
#     lr, step = config.lr_and_step(
#         [lsac.train.loss, lsac.val.loss] if isinstance(config, _ExpConfWithValid) else lsac.train.loss,
#         tf.train.get_or_create_global_step(), iters_per_epoch)
#
#     suppliers = plcs.create_suppliers([config.bs, config.bs, None])
#     tf.global_variables_initializer().run()
#
#     es = early_stopping(
#         config.pat, iters_per_epoch * config.epo, on_accept=lambda accept_iters, accept_val_acc:
#         update_append(statistics, es_accept=(accept_iters, accept_val_acc, lsac.test.acc.eval(suppliers.test())))
#     )
#     val_s2_size = min(timit.validation.num_examples, 50000)
#     val_supplier_2 = timit.validation.create_supplier(
#         plcs.plcs.val.x, plcs.plcs.val.y, batch_size=val_s2_size)
#
#     start_time = time.time()
#     for i in es:
#         if isinstance(config, TimitExpConfigHD):
#             ti = suppliers.train(max(i, i - 1))  # use ''validation'' placeholder for train
#             fd = merge_dicts(suppliers.train(i),
#                              {plcs.plcs.val.x: ti[plcs.plcs.train.x],
#                               plcs.plcs.val.y: ti[plcs.plcs.train.y]})
#         elif isinstance(config, _ExpConfWithValid):
#             fd = merge_dicts(suppliers.train(i), suppliers.val(i))
#         else:
#             fd = suppliers.train(i)
#         step(fd)  # for the baseline the validation supplier doesn't matter!
#
#         learning_rate = ss.run(lr)
#         update_append(statistics, learning_rate=learning_rate)
#         if i % int(config.cke * iters_per_epoch) == 0:
#             # compute full validation accuracy
#             validation_accuracy = np.mean([
#                 ss.run(lsac.val.acc, val_supplier_2(i))
#                 for i in range(timit.validation.num_examples // val_s2_size)
#             ])
#
#             test_accuracy = lsac.test.acc.eval(suppliers.test())
#             update_append(statistics,
#                           validation_accuracy=validation_accuracy,
#                           test_accuracy=test_accuracy, elapsed_time=str(timedelta(seconds=time.time() - start_time)),
#                           elapsed_time_sec=time.time() - start_time)
#             print(i, '\t', validation_accuracy, '\t', test_accuracy, '\t', learning_rate)
#             try:
#                 es.send(validation_accuracy)
#             except StopIteration:
#                 pass
#             gz_write(statistics, config.str_for_filename())
#
#     # end
#     end_string = '{} \t {} .'.format(time.asctime(), config.str_for_filename()) + \
#                  '\n iters, valid, test = {}  \t| total time {} \n'.format(
#                      statistics['es_accept'][-1], statistics['elapsed_time'][-1]
#                  )
#     with open('ledger.txt', 'a+') as f:
#         f.writelines(end_string)


if __name__ == '__main__':
    for name in UCI_DATASET_NAMES:
        print(name)
        data = load(name)
        print(data.train.num_examples,
              data.validation.num_examples,
              data.test.num_examples)

        print()

        print(data.train.dim_data, data.train.dim_target)

        print(data.train.data, data.train.target)
        print('-'*100)
