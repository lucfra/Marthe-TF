import gpu_manager

gpu_manager.setup_one_gpu()

from marthe import *
import tensorflow as tf
import far_ho.examples.datasets as dts

import intervaltree as it

import pandas as pd

from marthe import AllPlaceholders

TIMIT_DIR = '/home/common/DATASETS/timit4python'


def pad(_example, _size): return np.concatenate([_example] * _size)


class WindowedData(object):
    def __init__(self, data, row_sentence_bounds, window=5, process_all=False):
        """
        Class for managing windowed input data (like TIMIT).

        :param data: Numpy matrix. Each row should be an example data
        :param row_sentence_bounds:  Numpy matrix with bounds for padding. TODO add default NONE
        :param window: half-window size
        :param process_all: (default False) if True adds context to all data at object initialization.
                            Otherwise the windowed data is created in runtime.
        """
        self.window = window
        self.data = data
        base_shape = self.data.shape
        self.shape = (base_shape[0], (2 * self.window + 1) * base_shape[1])
        self.tree = it.IntervalTree([it.Interval(int(e[0]), int(e[1]) + 1) for e in row_sentence_bounds])
        if process_all:
            print('adding context to all the dataset', end='- ')
            self.data = self.generate_all()
            print('DONE')
        self.process_all = process_all

    def generate_all(self):
        return self[:]

    def __getitem__(self, item):  # TODO should be right for all the common use... But better write down a TestCase
        if hasattr(self, 'process_all') and self.process_all:  # keep attr check!
            return self.data[item]
        if isinstance(item, int):
            return self.get_context(item=item)
        if isinstance(item, tuple):
            if len(item) == 2:
                rows, columns = item
                if isinstance(rows, int) and isinstance(columns, int):  # TODO check here
                    # do you want the particular element?
                    return self.get_context(item=rows)[columns]
            else:
                raise TypeError('NOT IMPLEMENTED <|>')
            if isinstance(rows, slice):
                rows = range(*rows.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in rows])[:, columns]
        else:
            if isinstance(item, slice):
                item = range(*item.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in item])

    def __len__(self):
        return self.shape[0]

    def get_context(self, item):
        interval = list(self.tree[item])[0]
        # print(interval)
        left, right = interval[0], interval[1]
        left_pad = max(self.window + left - item, 0)
        right_pad = max(0, self.window - min(right, len(self) - 1) + item)  # this is to cope with reduce datasets
        # print(left, right, item)

        # print(left_pad, right_pad)
        base = np.concatenate(self.data[item - self.window + left_pad: item + self.window + 1 - right_pad])
        if left_pad:
            base = np.concatenate([pad(self.data[item], left_pad), base])
        if right_pad:
            base = np.concatenate([base, pad(self.data[item], right_pad)])
        return base


def load_timit(folder=TIMIT_DIR, only_primary=True, small=False, context=None,
               fake=False, process_all=False):
    def load_timit_sentence_bound():
        def sentence_bound_reader(name):
            bnd = pd.read_csv(folder + '/timit_%sSentenceBound.csv' % name, header=None).values
            return bnd - 1

        return [sentence_bound_reader(n) for n in ['train', 'val', 'test']]

    folder = folder or TIMIT_DIR
    if isinstance(process_all, bool):
        process_all = [process_all] * 3

    if fake:
        def generate_dataset(secondary=False):
            target = np.random.randn(2000, 183)
            if secondary:
                target = np.hstack([target, np.random.randn(2000, 300)])
            return np.random.randn(2000, 123), target

        training_data, training_target = generate_dataset(not only_primary)
        validation_data, validation_target = generate_dataset()
        test_data, test_target = generate_dataset()
        training_info_dict = None
    else:
        split_number = '00' if small else ''
        training_target = pd.read_csv(folder + '/timit_trainTargets%s.csv' % split_number, header=None).values
        training_data = pd.read_csv(folder + '/timit-preproc_traindata_norm_noctx%s.csv' %
                                    split_number, header=None).values
        training_info_dict = {'dim_primary_target': training_target.shape[1]}
        print('loaded primary training data')
        if not only_primary:
            training_secondary_target = pd.read_csv(folder + '/timit_trainTargetsPE%s.csv'
                                                    % split_number, header=None).values
            training_target = np.hstack([training_target, training_secondary_target])
            training_info_dict['dim_secondary_target'] = training_secondary_target.shape[1]
            print('loaded secondary task targets')

        validation_data = pd.read_csv(folder + '/timit-preproc_valdata_norm_noctx%s.csv'
                                      % split_number, header=None).values
        validation_target = pd.read_csv(folder + '/timit_valTargets%s.csv' % split_number, header=None).values
        print('loaded validation data')

        test_data = pd.read_csv(folder + '/timit-preproc_testdata_norm_noctx.csv', header=None).values
        test_target = pd.read_csv(folder + '/timit_testTargets.csv', header=None).values
        print('loaded test data')

    if context:
        sbs = load_timit_sentence_bound()
        training_data, validation_data, test_data = (WindowedData(d, s, context).generate_all() for d, s, pa
                                                     in zip([training_data, validation_data, test_data],
                                                            sbs, process_all))

    test_dataset = dts.Dataset(data=test_data, target=test_target)
    validation_dataset = dts.Dataset(data=validation_data, target=validation_target)
    training_dataset = dts.Dataset(data=training_data, target=training_target, info=training_info_dict)

    res = dts.Datasets(train=training_dataset, validation=validation_dataset, test=test_dataset)

    return res


def timit_ffnn(depth=4, units=2000, activation=None):
    if activation is None: activation = tf.nn.relu
    return [tf.layers.Dense(units=units, activation=activation) for k in range(depth - 1)] + [
        tf.layers.Dense(units=183, activation=None)
    ]


class TimitExpConfig(Config):

    def __init__(self, **kwargs):
        self.lr0 = 0.075
        self.momentum = 0.5
        self.train_bs = 128
        self.max_epochs = 15
        self.check_every = 100
        self.patience = 20
        self.seed = 1
        self.context = 5
        self.small_dataset = False
        super().__init__(**kwargs)

    def lr_and_step(self, loss):
        step = tf.train.MomentumOptimizer(self.lr0, self.momentum).minimize(loss)
        return self.lr0, lambda fd: step.run(fd)


class TimitExpConfigMarthe(TimitExpConfig):

    def __init__(self, **kwargs):
        self.beta = 1.e-6
        super().__init__(**kwargs)

    def lr_and_step(self, loss):
        lr = get_positive_hyperparameter('lr', self.lr0)

        optimizer = MomentumOptimizer(lr, self.momentum)

        opt_dict = optimizer.minimize(loss[0])
        marthe = Marthe(beta=self.beta)
        marthe.compute_gradients(loss[1], opt_dict)
        return lr, marthe.run


def timit_exp(config: TimitExpConfig):
    ss = setup_tf(config.seed)

    timit = load_timit(only_primary=True, context=config.context,
                       small=config.small_dataset)
    plcs = AllPlaceholders(timit)

    model = timit_ffnn()
    outs = plcs.build_recursive_outputs(model)
    lsac = plcs.losses_and_accs(outs)

    lr, step = config.lr_and_step([lsac.train.loss, lsac.val.loss]
                       if isinstance(config, TimitExpConfigMarthe)
                       else lsac.train.loss)

    suppliers = plcs.create_suppliers([config.train_bs, config.train_bs, None])
    tf.global_variables_initializer().run()

    iters = timit.train.num_examples//config.train_bs * config.max_epochs
    print('TOTAL NUMBER OF ITERATIONS', iters)

    for i in range(iters):
        merged_exs = merge_dicts(suppliers.train(i), suppliers.val(i))
        step(merged_exs)

        if i % config.check_every == 0:
            eva = ss.run(lsac.val.acc, suppliers.val(-1))
            etst = lsac.test.acc.eval(suppliers.test())
            lrv = ss.run(lr)
            print(i, '\t', eva, '\t', etst, '\t', lrv)


if __name__ == '__main__':
    timit_exp(TimitExpConfigMarthe())
    # marthe_timit(beta=1.e-5)
    # baseline()

    # timit = load_timit(folder='/home/common/DATASETS/timit4python',only_primary=True, context=5)
    # print(timit)
    #
    # print(timit.train)
    #
    # print(timit.validation)
    #
    # print(timit.test)
    #
    # x, y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
    #
    # bs_gen = timit.train.create_supplier(x, y, batch_size=100)
    #
    # print(bs_gen(0))
    #
    # print(bs_gen(1))
    #
    # print(timit.train.dim_data, timit.train.dim_target)