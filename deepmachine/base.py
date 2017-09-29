import tensorflow as tf
import warnings
import os
import functools

from pathlib import Path

slim = tf.contrib.slim

from .flags import FLAGS
from . import ops
from . import summary


def _undefined_op(*args, **kwargs):
    raise NotImplementedError


class DeepMachine(object):
    """docstring for DeepMachine"""

    def __init__(self,
                 network_op,
                 restore_path=None,
                 summary_ops=[],
                 losses_ops=[],
                 **kargs
                 ):
        # config
        tf.logging.set_verbosity(FLAGS.logging_level)

        self._config = tf.ConfigProto(
            allow_soft_placement=True
        )

        # dynamic ops
        self._loss_ops = losses_ops
        self._summary_ops = summary_ops

        self.add_summary_op(summary.summary_input)
        self.add_summary_op(summary.summary_total_loss)
        self.add_summary_op(summary.summary_predictions)

        # cached objects
        self._sess = None
        self._train_graph = None
        self._run_graph = None
        self._test_graph = None

        # configurable ops
        self.network_op = network_op
        self.train_op = ops.train.adam
        self.init_op = ops.init.restore
        self.eval_op = _undefined_op
        self.restore_path = restore_path
        self.pre_process_fn = tf.identity

    @property
    def network_op(self):
        return self.__network_op

    @network_op.setter
    def network_op(self, value):
        self.__network_op = value
        self._reset_graph()

    @property
    def restore_path(self):
        return self.__restore_path

    @restore_path.setter
    def restore_path(self, value):

        if value is not None:

            path = Path(value)

            if path.is_dir():
                value = tf.train.latest_checkpoint(value)

        self.__restore_path = value
        self._reset_graph()

    @property
    def train_op(self):
        return self.__train_op

    @train_op.setter
    def train_op(self, value):
        self.__train_op = value
        self._reset_graph()

    @property
    def init_op(self):
        return self.__init_op

    @init_op.setter
    def init_op(self, value):
        self.__init_op = value
        self._reset_graph()

    @property
    def eval_op(self):
        return self.__eval_op

    @eval_op.setter
    def eval_op(self, value):
        self.__eval_op = value
        self._reset_graph()
        
    @property
    def pre_process_fn(self):
        return self.__pre_process_fn

    @pre_process_fn.setter
    def pre_process_fn(self, value):
        self.__pre_process_fn = value
        self._reset_graph()

    def add_loss_op(self, op):
        self._loss_ops.append(op)

    def add_summary_op(self, op):
        self._summary_ops.append(op)

    def train(self,
              train_data_op,
              db_size,
              saver=None,
              train_dir=FLAGS.train_dir,
              log_every_n_steps=FLAGS.log_every_n_steps,
              initial_learning_rate=FLAGS.initial_learning_rate * FLAGS.batch_size,
              batch_size=FLAGS.batch_size,
              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
              moving_average_ckpt=FLAGS.moving_average_ckpt,
              number_of_epochs=FLAGS.number_of_epochs
              ):

        # parameters
        number_of_steps = number_of_epochs * db_size // batch_size
        learning_rate_decay_step = db_size // batch_size

        if number_of_steps == 0:
            number_of_steps = None

        # build graph if needed
        g = tf.Graph()
        with g.as_default():

            # Load datasets.
            data_eps = train_data_op()

            # Define model graph.
            net_eps = self.network_op(
                data_eps['inputs'], is_training=True, data_eps=data_eps)

            # custom losses
            if len(self._loss_ops) == 0:
                warnings.warn(
                    'No loss function defined. The training progress will not have any effect')
            for op in self._loss_ops:
                op(data_eps, net_eps)

            # global_step
            self._global_step = slim.get_or_create_global_step()

            # train_op
            self._train_ops = self.train_op(
                initial_learning_rate=initial_learning_rate,
                learning_rate_decay_step=learning_rate_decay_step,
                batch_size=batch_size,
                learning_rate_decay_factor=learning_rate_decay_factor,
                moving_average_ckpt=moving_average_ckpt
            )

            if type(self._train_ops) is list:
                if not number_of_steps is None:
                    number_of_steps *= len(self._train_ops)
                self._train_ops = sum(self._train_ops)

            print('Total # of Steps: %d' % number_of_steps)

            # summaries
            for op in self._summary_ops:
                op(data_eps, net_eps, is_training=True)

        self._train_graph = g

        # start session
        with self._train_graph.as_default():
            with tf.Session(graph=self._train_graph,
                            config=self._config) as sess:

                # init and start
                init_fn = self.init_op()

                slim.learning.train(self._train_ops,
                                    train_dir,
                                    save_summaries_secs=60,
                                    init_fn=init_fn,
                                    global_step=self._global_step,
                                    number_of_steps=number_of_steps,
                                    save_interval_secs=600,
                                    saver=saver,
                                    log_every_n_steps=log_every_n_steps)

    def test(self,
             eval_data_op=_undefined_op,
             eval_dir=FLAGS.eval_dir,
             train_dir=FLAGS.train_dir,
             eval_size=FLAGS.eval_size):

        g = tf.Graph()
        with g.as_default():

            # Load datasets.
            data_eps = eval_data_op()

            # Define model graph.
            net_eps = self.network_op(data_eps['inputs'], is_training=False, data_eps=data_eps)

        self._test_graph = g

        with self._test_graph.as_default():
            with tf.Session(graph=self._test_graph) as sess:

                eval_ops, summary_ops = self.eval_op(data_eps, net_eps)

                slim.evaluation.evaluation_loop(
                    '',
                    train_dir,
                    eval_dir,
                    num_evals=eval_size,
                    eval_op=eval_ops,
                    summary_op=tf.summary.merge(summary_ops),
                    eval_interval_secs=30)

    def run_one(self, data, dtype=tf.float32, feed_dict=feed_dict):
        return self._run(data, dtype=dtype, feed_dict=feed_dict)

    def run_batch(self, data, dtype=tf.float32, feed_dict=feed_dict):
        return self._run(data, dtype=dtype, feed_dict=feed_dict)

    def _run(self, data, dtype, feed_dict={}):
        # build graph if needed
        if self._run_graph is None:
            with tf.Graph().as_default() as g:

                # inputs placeholder
                tfinputs = tf.placeholder(
                    dtype,
                    shape=(None, None, None, data.shape[-1]),
                    name='inputs'
                )
                
                tfinputs = self.pre_process_fn(tfinputs)

                # Define model graph.
                self._run_net_eps = self.network_op(
                    tfinputs, is_training=False, **kwargs)

                self._run_saver = None
                variables_to_restore = slim.get_variables_to_restore()
                if variables_to_restore:
                    self._run_saver = tf.train.Saver(variables_to_restore)

            self._run_graph = g

        # start session
        sess, need_restart = self._get_session(
            self._run_graph,
            return_restart=True)

        if need_restart and self._run_saver:
            self._run_saver.restore(sess, self.restore_path)
            
        feed_dict['inputs:0'] = data

        return sess.run(self._run_net_eps, feed_dict=feed_dict)

    def _get_session(self, graph, return_restart=False):
        restart = False

        if self._sess is None:
            restart = True
        else:
            if self._sess.graph is not self._run_graph:
                self._sess.close()
                restart = True

        if restart:
            self._sess = tf.Session(graph=graph)

        if return_restart:
            return self._sess, restart

        return self._sess

    def _reset_graph(self):
        self._train_graph = None
        self._run_graph = None
        self._test_graph = None
