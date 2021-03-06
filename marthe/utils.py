import gzip
import pickle
from collections import OrderedDict, Iterator
from functools import reduce

import tensorflow as tf
import numpy as np

import marthe as mt

import time


def dot(a, b, name=None):
    """
    Dot product between vectors `a` and `b` with optional name.
    If a and b are not vectors, formally this computes <vec(a), vec(b)>.
    """
    # assert a.shape.ndims == 1, '{} must be a vector'.format(a)
    # assert b.shape.ndims == 1, '{} must be a vector'.format(b)
    with tf.name_scope(name, 'Dot', [a, b]):
        return tf.reduce_sum(a*b)


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, 'Vectorization', var_list) as scope:
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0, name=scope)


# noinspection PyClassHasNoInit
class GraphKeys(tf.GraphKeys):
    """
    adds some hyperparameters and hypergradients computation related keys
    """

    HYPERPARAMETERS = 'hyperparameters'


def hyperparameters(scope=None):
    """
    List of variables in the collection HYPERPARAMETERS.

    Hyperparameters constructed with `get_hyperparameter` are in this collection by default.

    :param scope: (str) an optional scope.
    :return: A list of tensors (usually variables)
    """
    return tf.get_collection(GraphKeys.HYPERPARAMETERS, scope=scope)


HYPERPARAMETERS_COLLECTIONS = [GraphKeys.HYPERPARAMETERS, GraphKeys.GLOBAL_VARIABLES]


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.
    """
    return obj if isinstance(obj, list) else [obj]


# noinspection PyArgumentList,PyTypeChecker
def get_hyperparameter(name, initializer=None, shape=None, dtype=None, collections=None,
                       scalar=False, constraint=None):
    """
    Creates an hyperparameter variable, which is a GLOBAL_VARIABLE
    and HYPERPARAMETER. Mirrors the behavior of `tf.get_variable`.

    :param name: name of this hyperparameter
    :param initializer: initializer or initial value (can be also np.array or float)
    :param shape: optional shape, may be not needed depending on initializer
    :param dtype: optional type,  may be not needed depending on initializer
    :param collections: optional additional collection or list of collections, which will be added to
                        HYPERPARAMETER and GLOBAL_VARIABLES
    :param scalar: default False, if True splits the hyperparameter in its scalar components, i.e. each component
                    will be a single scalar hyperparameter. In this case the method returns a tensor which of the
                    desired shape (use this option with `ForwardHG`)
    :param constraint: optional contstraint for the variable (only if not scalar..)

    :return: the newly created variable, or, if `scalar` is `True` a tensor composed by scalar variables.
    """
    _coll = list(HYPERPARAMETERS_COLLECTIONS)
    if collections:
        _coll = _coll + as_list(collections)  # this might not work.... bah
    if not scalar:
        try:
            return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                                   collections=_coll,
                                   constraint=constraint)
        except TypeError as e:
            print(e)
            print('Trying to ignore constraints (to use constraints update tensorflow.')
            return tf.get_variable(name, shape, dtype, initializer, trainable=False,
                                   collections=_coll)
    else:
        with tf.variable_scope(name + '_components'):
            _shape = shape or initializer.shape
            if isinstance(_shape, tf.TensorShape):
                _shape = _shape.as_list()
            _tmp_lst = np.empty(_shape, object)
            for k in range(np.multiply.reduce(_shape)):
                indices = np.unravel_index(k, _shape)
                _ind_name = '_'.join([str(ind) for ind in indices])
                _tmp_lst[indices] = tf.get_variable(_ind_name, (), dtype,
                                                    initializer if callable(initializer) else initializer[indices],
                                                    trainable=False, collections=_coll)
        return tf.convert_to_tensor(_tmp_lst.tolist(), name=name)


def get_hyper_box_constraints(name, value, minval, maxval, **kwargs):
    return get_hyperparameter(
        name, value,
        constraint=lambda t: tf.maximum(minval, tf.minimum(maxval, t)), **kwargs
    )


def get_positive_hyperparameter(name, value, **kwargs):
    return get_hyperparameter(
        name, value,
        constraint=lambda t: tf.maximum(0., t), **kwargs
    )


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def merge_dicts(*dicts):
    """
    Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
    """
    from functools import reduce
    # if len(dicts) == 1: return dicts[0]
    return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


def build_recursive_model(x, layers):
    return reduce(lambda tensor, new_layer: new_layer(tensor), layers, x)


def loss_and_acc(out, y, regression=False):
    if regression:
        loss = tf.reduce_mean(tf.squared_difference(out, tf.cast(y, out.dtype)))
        accuracy = - loss  # to keep the conformity with early stopping and so on
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')
    return loss, accuracy


class Placeholders:

    def __init__(self, dim_x=28 * 28, dim_y=10, name='train'):
        with tf.name_scope(name):
            dim_x = mt.as_list(dim_x)
            self.x = tf.placeholder(tf.float32, shape=[None] + dim_x, name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, dim_y], name='y')


class TVT(list):  # train val & test

    def __init__(self, lst) -> None:
        super().__init__(lst)
        assert len(lst) == 3
        self.train, self.val, self.test = lst


class LA:  # loss and accuracy

    def __init__(self, la):
        self.loss, self.acc = la


class AllPlaceholders:

    def __init__(self, datasets):
        self.datasets = datasets
        self.plcs = TVT([Placeholders(d.dim_data, d.dim_target) for d in datasets])

    def create_suppliers(self, batch_sizes):
        return TVT([d.create_supplier(p.x, p.y, batch_size=bs) for
                    d, p, bs in zip(self.datasets, self.plcs, batch_sizes)])

    def build_recursive_outputs(self, layers):
        return TVT([build_recursive_model(p.x, layers) for p in self.plcs])

    def losses_and_accs(self, outs):
        return TVT([LA(loss_and_acc(o, p.y)) for o, p in zip(outs, self.plcs)])


def update_append(dct, **updates):  # for saving statistics
    for k, e in updates.items():
        dct[k].append(e)


def gz_read(name):
    name = '{}.gz'.format(name)
    with gzip.open(name, 'rb') as f:
        return pickle.load(f)


def gz_write(content, name):
    name = '{}.gz'.format(name)
    with gzip.open(name, 'wb') as f:
        pickle.dump(content, f)


def setup_tf(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    # noinspection PyTypeChecker
    np.random.set_state(np.random.RandomState(seed).get_state())
    if tf.get_default_session(): tf.get_default_session().close()
    return tf.InteractiveSession()


def early_stopping(patience, maxiters=1e10, on_accept=None, on_refuse=None, on_close=None, verbose=True):
    """
    Generator that implements early stopping. Use `send` method to give to update the state of the generator
    (e.g. with last validation accuracy)

    :param patience:
    :param maxiters:
    :param on_accept: function to be executed upon acceptance of the iteration
    :param on_refuse: function to be executed when the iteration is rejected (i.e. the value is lower then best)
    :param on_close: function to be exectued when early stopping activates
    :param verbose:
    :return: a step generator
    """
    val = None
    pat = patience
    t = 0
    while pat and t < maxiters:
        new_val = yield t
        if new_val is not None:
            if val is None or new_val > val:
                val = new_val
                pat = patience
                if on_accept:
                    try:
                        on_accept(t, val)
                    except TypeError:
                        try:
                            on_accept(t)
                        except TypeError:
                            on_accept()
                if verbose: print('ES t={}: Increased val accuracy: {}'.format(t, val))
            else:
                pat -= 1
                if on_refuse: on_refuse(t)
        else:
            t += 1
    yield t  # not sure this is necessary
    if on_close: on_close(val)
    if verbose: print('ES: ending after', t, 'iterations', 'patience=', pat)


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self.vrs = 3  # version
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[\n' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items()) if not k.startswith('_')) + ']\n'

    def str_for_filename(self):
        name = str(self)
        return name.replace('\n', ' ').replace('\t', '')
        # name.replace('\t', '')

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]

    @classmethod
    def random(cls, budget, the_seed, **kwargs):
        """
        Perform random search. If the value of a config parameter is not a callable, then it is considered fixed"

        :param budget: should be formated in hh:mm:ss  (string)
        :param the_seed: random seed for the random search, the random state is passed as argument of the callable
        :param kwargs: the configuarion parameters, followed by a callable
        :return: an iterator
        """
        rnd = np.random.RandomState(the_seed)
        start_time = time.time()
        total_time = sum(x * int(t) for x, t in zip([1, 60, 3600], reversed(budget.split(":"))))
        while time.time() - start_time < total_time:
            sin = OrderedDict({k: v for k, v in kwargs.items() if callable(v)})
            yield cls(**merge_dicts(
                {k: v for k, v in kwargs.items() if not callable(v)},
                {k: v(rnd) for k, v in sin.items()})
                )


def run_method_on_configs(method, config):
    if isinstance(config, Iterator) or isinstance(config, list):
        rss = OrderedDict()
        for c in config:
            setup_tf(c.sd if hasattr(c, 'sd') else None)
            rss[c] = method(c)
        return rss
    else: return method(config)