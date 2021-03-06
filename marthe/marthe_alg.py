# from far_ho.hyper_gradients import *
from marthe.optimizers import *
import tensorflow as tf
from marthe import utils
from tensorflow.python.training import slot_creator
import numpy as np

_ERROR_NOT_OPTIMIZER_DICT = """
    Looks like {} is not an `OptimizerDict`. Use optimizers in far_ho.optimizers for obtaining an OptimizerDict.
    """


def grad(ys, xs):
    try:
        # noinspection PyUnresolvedReferences
        return tf.gradients(ys, xs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    except AttributeError:  # previous versions of tensorflow....
        grads = tf.gradients(ys, xs)
        return [g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, xs if isinstance(xs, list) else [xs])]


def clip_and_count(t, counter, clip_val):
    """

    :param t: tensor to clip
    :param counter: integer variable; counts how many times clip is performed.
    :param clip_val: value
    :return: clipped t
    """

    def clip_and_increase():
        with tf.control_dependencies([counter.assign_add(1)]):
            return tf.clip_by_value(t, -clip_val, clip_val)

    if clip_val:
        return tf.cond(tf.not_equal(t, tf.clip_by_value(t, -clip_val, clip_val)),
                       lambda: clip_and_increase(), lambda: t)
    else:
        return t


class Marthe:

    def __init__(self, outer_obj_optimizer: tf.train.Optimizer=None, mu='adapt', name='Marthe', gs=None, beta=None):
        # TODO make it more similar to tf.train.Optimizer
        """

        :param outer_obj_optimizer: an optimizer for the learning rate (default GradientDescent)
                                      better not to use methods with momentum since they may overshoot the learning
                                      rate causing divergence.
        :param mu: float or `adapt`. If float uses a fixed dumping factor, otherwise adapts it online as described
                            in the paper
        :param name: Name of this object
        :param gs:  An optional global step
        :param beta: This is the only real parameter to set. Good values seems to range between [1.e-15 and 1.e-6].
                        It should be easier to set this beta rather than the hyper-learning rate.
        """
        self.name = name
        if outer_obj_optimizer is None:
            outer_obj_optimizer = tf.train.GradientDescentOptimizer(tf.placeholder(tf.float32, name='beta_prime'))
        self._outer_object_optimizer = outer_obj_optimizer

        self._w_dots = []
        self._w_dots_iterations = []

        self.mu = mu

        self._hypergrads = []
        self.hypergrads = {}

        self._hyper_list = self._iteration = None  # TODO find a better way

        self.mu_pl = tf.placeholder(tf.float32)  # TODO better implementation?
        self.mu_val = self.mu if isinstance(self.mu, float) else 0.
        self._max_q = 1.
        # self.assign_mu_v_p

        self.gs = gs
        self.Bs = []
        self.vec_outer_obj_grads = self._opt_dict = self.step = None
        self.c = 0.  # this doesn't matter

        self.beta = self._outer_object_optimizer._learning_rate   # used only if alpha is not None
        self.beta_val, self.prev_delta = 1.e-8, 0.

        # if beta == 'auto':
        #     self.alpha = (1.e-2, 10.)  # initial value and decrease coefficient
        # else:
        self.alpha = beta

        self.hg_clip_counter = tf.Variable(0, trainable=False, name='hg_clip_counter')  # TODO ... SORRY FOR THIS
        self.alpha_clip_counter = 0

        self._delta = 0.

    def compute_gradients(self, outer_objective, optimizer_dict: OptimizerDict,
                          hyper_list=None, clip_value=100.):
        """
        This methods populates the computational graph

        :param outer_objective:  optimization objective for the learning rate (e.g. validation error)
        :param optimizer_dict: on `OptimizerDict` (see marthe.optimizers) e.g.
                                 opt_dict = marthe.GradientDescentOptimizer(lr).minimize(training_error)
        :param hyper_list:   an optional list of hyperparameters (the learning rate). By default all variables
                                in the collection `HYPERPARAMETERS`  are taken
        :param clip_value: optional value for clipping the Marthe lr update
        """
        assert isinstance(optimizer_dict, OptimizerDict), _ERROR_NOT_OPTIMIZER_DICT.format(optimizer_dict)
        self._opt_dict = optimizer_dict

        if hyper_list is None:  # get default hyperparameters
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)

        state = list(optimizer_dict.state)
        print('HG - COMPUTE GRADIENTS - LEN STATE: {}'.format(len(state)))

        vs = [tf.ones_like(w) for w in state]  # `ghost variables'

        vec_vs = utils.vectorize_all(vs)
        print('HG - COMPUTE GRADIENTS - TOTAL PARAMETERS: {}'.format(vec_vs.get_shape().as_list()))
        dynamics = list(optimizer_dict.dynamics)
        vec_dynamics = utils.vectorize_all(dynamics)

        outer_obj_grads = grad(outer_objective, state)
        self.vec_outer_obj_grads = utils.vectorize_all(outer_obj_grads)  # also used for heuristics

        self._w_dots = [[slot_creator.create_zeros_slot(
            w, name='w_dot_{}'.format(h.op.name)) for w in state]
            for h in hyper_list]
        vec_w_dots = [utils.vectorize_all(w_dot) for w_dot in self._w_dots]

        for hyper, w_dot, vec_w_dot in zip(hyper_list, self._w_dots, vec_w_dots):
            assert hyper.shape.ndims == 0

            A_w_dot = grad(utils.dot(vec_dynamics, vec_w_dot), state)
            B = grad(grad(utils.dot(vec_dynamics, vec_vs), hyper)[0], vs)

            self.Bs.append(utils.vectorize_all(B))  # used in the heuristics

            mu = tf.convert_to_tensor(self.mu_pl, dtype=A_w_dot[0].dtype)

            if self.mu == 'adapt' or self.mu > 0:
                self._w_dots_iterations.append(
                    [wd.assign(mu * awd + b) for wd, awd, b in zip(w_dot, A_w_dot, B)])
            else:
                self._w_dots_iterations.append(
                    [wd.assign(b) for wd, awd, b in zip(w_dot, A_w_dot, B)])

            hg = clip_and_count(utils.dot(self.vec_outer_obj_grads, vec_w_dot), self.hg_clip_counter, clip_value)
            # tf.add_to_collection(hg, utils.GraphKeys.HYPERGRADIENTS)

            # todo ADD d E / d lambda when required
            self._hypergrads.append(hg)
            self.hypergrads[hyper] = hg
            self._hyper_list = hyper_list

        def _apply_hg():
            print(self._hypergrads, self._hyper_list)
            return self._outer_object_optimizer.apply_gradients(list(
                zip(self._hypergrads, self._hyper_list)), global_step=self.gs)

        with tf.control_dependencies([_apply_hg()]):
            with tf.control_dependencies(self._w_dots_iterations[0]):
                self.step = self._opt_dict.iteration  # hopefully this still must be compiled... otherwise with these
                # bloody dependencies everything goes to shit

    def run(self, fd=None, clip_alpha=None):
        """
        Executes an iteration of the method

        :param fd: optional dictionary of feeds for performing one step of training error optimization and one step
                    of learning rate update. Attention: inner and outer objective computations are carried out
                    simultaneously: use two different output tensors and relative feeds.
        :param clip_alpha: optional clip value for the update of the hyper-learning rate
        """
        ss = tf.get_default_session()

        # optimization step and hypergradient step. Gets the hypergradient and the [B_t]_t vector (negative gradient
        # if the training error for sgd)
        # here it is assumed that there are 2 outputs (one on the training set, the other on the validation set
        # the order of computation is:
        #
        # \eta_t = \eta_{t-1} - \beta \delta
        # Z_t+1 = ...
        # w_t+1 = ...
        #

        dct = utils.merge_dicts(fd, {self.mu_pl: self.mu_val})

        if self.alpha is not None:  # crappy heuristics for beta,
            # delta = self._hypergrads[-1].eval(dct)
            upd_beta = self._delta*self.prev_delta
            # print(self._delta, self.prev_delta)
            if clip_alpha: upd_beta = np.max([np.min([upd_beta, clip_alpha]), -clip_alpha])
            self.beta_val = np.max([self.beta_val + self.alpha * upd_beta, 0.])
            self.prev_delta = np.array(self._delta)
            dct[self.beta] = self.beta_val  # update dictionary

        if self.mu == 'adapt' or self.alpha is not None:
            delta, b_t = ss.run([self._hypergrads[0], self.Bs[0]], dct)  # only for 1 hyper
            # print('CHECK THIS', self._hyper_list[0].eval(), '\t', self.beta_val, '\t', delta)
            self._delta = delta  # save for updating mu and beta
        # b_t is for mu
        ss.run(self.step, dct)  # even though these 2 instructions could be run togheter.. you never know with tf....

        # print('after:', self._hyper_list[0].eval())

        if self.mu == 'adapt':
            e_t_1 = ss.run(self.vec_outer_obj_grads, fd)  # outer_objective grad  \nabla E(w_{t+1})
            _1st_ord_cond = np.dot(e_t_1, b_t)  # scalar product: for sgd:  -\nabla L(w_t) \cdot \nabla E(w_{t+1}

            # IMPORTANT, validation and training loss grads should be computed at different steps!
            q_norm = self._delta/ _1st_ord_cond  # normalization

            z = np.maximum(np.minimum(q_norm, 1.), 0)  # clipping between 0, and 1

            self.c = (self.c + 1)*self.mu_val
            self.mu_val = np.power(z, 1. / (self.c + 1.))
