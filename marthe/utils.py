import tensorflow as tf
import numpy as np


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