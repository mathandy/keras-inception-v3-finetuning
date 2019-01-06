from __future__ import division, print_function, absolute_import
from six.moves import zip
import tensorflow as tf
import numpy as np
from andnn.losses import ce_wlogits
from andnn.utils import step_plot, accuracy, num_correct, num_incorrect, batches
import time
from andnn.utils import Timer, ppercent, pnumber


def get_batch(data, batch_size, step):
    offset = (step * batch_size) % (data.shape[0] - batch_size)
    return data[offset:(offset + batch_size)]


def uninintialized_variable_initializer(session):
    uninitialized_var_names = session.run(tf.report_uninitialized_variables())
    print(uninitialized_var_names)
    uninitialized_vars = \
        [tf.get_variable(name) for name in uninitialized_var_names]
    return session.run(tf.variables_initializer(uninitialized_vars))


from tensorflow.core.framework import summary_pb2
def make_summary(name, val):
    return summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=name,
                                         simple_value=val)])


class AnDNNClassifier:
    def __init__(self, model, example_shape, label_shape, final_activation=None,
                 session=None, debug=False, tensorboard_dir='/tmp/tflogs'):
        """Note: if loss expects logits, then `model` should output logits and 
        `final_activation` should be used."""
        self.model = model
        self.example_shape = tuple(example_shape)
        self.label_shape = tuple(label_shape)
        self.final_activation = final_activation
        self.session = session
        self.debug = debug
        self.tb_dir = tensorboard_dir

        self._X = tf.placeholder(tf.float32, (None,) + self.example_shape)
        self._Y = tf.placeholder(tf.float32, (None,) + self.label_shape)

        if self.final_activation is None:
            self._predictions = self.model(self._X)
            # self._loss = loss(self._predictions, self._Y, **loss_kwargs)
        else:
            self._logits = self.model(self._X)
            self._predictions = self.final_activation(self._logits)
            # self._loss = loss(self._logits, self._Y, **loss_kwargs)



    # def load_weights(self, weights, np_load_kwargs={}):
    #     if isinstance(weights, str):
    #         weights = np.load(weights, **np_load_kwargs)
    #
    #     parameters = tf.trainable_variables()
    #     parameters = tf.Print(parameters, [parameters],
    #                           '\n\ntrainable_variables: ')
    #
    #     keys = sorted(weights.keys())
    #     for i, k in enumerate(keys):
    #         self._sess.run(self.parameters[i].assign(weights[k]))

    # def save_weights(self, weight_file, np_savez_kwargs):
    #     """UNTESTED/UNFINISHED"""
    #     keys = sorted(self.parameters)
    #     np.savez(weight_file, self.parameters, **np_savez_kwargs)
    #
    #     for i, k in enumerate(keys):
    #         self._sess.run(self.parameters[i].assign(weights[k]))

    # def get_fc_weight_reshape(self, name, shape, num_classes=None):
    #     print('Layer name: %s' % name)
    #     print('Layer shape: %s' % shape)
    #     weights = self.data_dict[name][0]
    #     weights = weights.reshape(shape)
    #     if num_classes is not None:
    #         weights = self._summary_reshape(weights, shape,
    #                                         num_new=num_classes)
    #     init = tf.constant_initializer(value=weights,
    #                                    dtype=tf.float32)
    #     var = tf.get_variable(name="weights", initializer=init, shape=shape)
    #     return var

    def create_checkpoint(self, step, checkpoint_name=None):
        self._saver.save(self._sess, checkpoint_name, global_step=step)

    def _initialize(self):



        # initialize session
        if self.session is None:
            self.session = tf.Session()
        if self.debug:
            from tensorflow.python import debug as tf_debug
            self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)

            def always_true(*vargs):
                return True

            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.session.add_tensor_filter("has_inf_or_nan", always_true)

        # initialize variables (if necessary)
        # self.session.run(tf.initialize_variables(list(tf.get_variable(name) for name in
        #                                       self.session.run(
        #                                           tf.report_uninitialized_variables(
        #                                               tf.all_variables())))))
        # uninintialized_variable_initializer(self.session)
        # self.session.run(uninintialized_variable_initializer())
        self.session.run(tf.global_variables_initializer())


    def _build_learning_mechanism(self, loss, loss_kwargs,
                                  optimizer, optimizer_kwargs):
        if self.final_activation is None:
            # self._predictions = self.model(self._X)
            self._loss = loss(self._predictions, self._Y, **loss_kwargs)
        else:
            # self._logits = self.model(self._X)
            # self._predictions = self.final_activation(self._logits)
            self._loss = loss(self._logits, self._Y, **loss_kwargs)
        tf.summary.scalar('loss', self._loss)

        # training
        self._train_op = optimizer(**optimizer_kwargs).minimize(self._loss)

        # scoring and summarizing
        last_axis = len(self.label_shape)
        is_correct = tf.equal(tf.argmax(self._predictions, axis=last_axis),
                              tf.argmax(self._Y, axis=last_axis))
        self._number_correct = tf.reduce_sum(tf.cast(is_correct, tf.float32))
        self._accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar('accuracy', self._accuracy)

        # prep for tensorboard
        self._summary = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter(self.tb_dir + '/retrain')
        self._val_writer = tf.summary.FileWriter(self.tb_dir + '/val')

        self._initialize()

        # add initialized graph to tensorboard
        self._train_writer.add_graph(self.session.graph)

        # construct saver object
        self.saver = tf.train.Saver()

    def fit(self, X_train, Y_train, batch_size, epochs, loss=ce_wlogits,
            loss_kwargs={}, optimizer=tf.train.AdamOptimizer,
            optimizer_kwargs={'learning_rate': 1e-5}, steps_per_report=500,
            report_first_n=20, X_valid=None, Y_valid=None,
            validation_batch_size=None, save_conditions=lambda s, a, l: False,
            savename='unnamed-'+str(time.time())+'-'):
        self._build_learning_mechanism(loss=loss,
                                       loss_kwargs=loss_kwargs,
                                       optimizer=optimizer,
                                       optimizer_kwargs=optimizer_kwargs)
        max_acc = 0
        step_accuracy = []
        step_loss = []
        # try:
        for step in range(epochs * Y_train.shape[0]):
            # get batch ready for training step
            # feed_dict = {X: next(X_batch), Y: next(Y_batch)}

            X_batch = get_batch(X_train, batch_size, step)
            Y_batch = get_batch(Y_train, batch_size, step)
            feed_dict = {self._X: X_batch,
                         self._Y: Y_batch}
            fetches = [self._train_op, self._loss, self._accuracy,
                       self._summary]

            atime = time.time()
            _, l, a, s = self.session.run(fetches, feed_dict)
            train_time = time.time() - atime

            # report to TensorBoard
            self._train_writer.add_summary(s, step)


            # validate and report through terminal
            max_acc = max(max_acc, a)
            step_accuracy.append(a)
            step_loss.append(l)
            if (step % steps_per_report) == 0 or step < report_first_n:
                if X_valid is not None:
                    atime = time.time()
                    vl, va = self.score(X_valid, Y_valid,
                                        validation_batch_size, False)
                    val_time = time.time() - atime

                    if save_conditions(step, vl, va):
                        saver.save(sess, save_name, global_step=step)

                    # vs = self.session.run([self._va_summary],
                    #                       {self._tb_record: va})
                    self._val_writer.add_summary(
                        make_summary("loss", vl), step)
                    self._val_writer.add_summary(
                        make_summary("accuracy", va), step)
                    # self.val_writer.add_summary(vs, step)


                    print("retrain ({}) | val ({})"
                          ":: {: 5.2%} | {: 5.2%} "
                          ":: {: 5.4G} | {: 5.4G} "
                          ":: step {} / {}"
                          "".format(Timer.format_time(train_time),
                                    Timer.format_time(val_time),
                                    ppercent(a), ppercent(va), pnumber(l),
                                    pnumber(vl), step,
                                    epochs * Y_train.shape[0]))

                else:
                    print("retrain | val "
                          ":: {: 6.2%} | N/A "
                          ":: {: 5.4G} | N/A "
                          ":: step {} / {}"
                          "".format(a, l, step,
                                    epochs * Y_train.shape[0]))

        # except KeyboardInterrupt:
        #     print("KEYBOARD INTERRUPT")
        # finally:
        #     print("max accuracy:", max_acc)
        #     step_plot([step_accuracy, step_loss],
        #               ['step_accuracy', 'step_loss'])

    def score(self, X, Y, batch_size, report=True):
        loss = 0
        num_correct = 0
        for X_batch, Y_batch in zip(batches(X, batch_size),
                                    batches(Y, batch_size)):
            # feed_dict = {self._X: X_batch, self._Y: Y_batch,
            #              self._previous_num_correct: num_correct}
            # fetches = [self._loss, self._num_correct_running_total]
            #
            # loss, num_correct = self.session.run(fetches, feed_dict)

            feed_dict = {self._X: X_batch, self._Y: Y_batch}
            fetches = [self._loss, self._number_correct]
            l, c = self.session.run(fetches, feed_dict)
            loss += l*len(X_batch)/len(X)
            num_correct += c
        percent_correct = num_correct/np.prod(Y.shape[:-1])
        if report:
            print("Validation: Loss = {:0.4G} | a={:06.2%}"
                  "".format(loss, percent_correct))
        return loss, percent_correct

    def transform(self, X, batch_size, session=tf.Session()):
        for batch in batches(X, batch_size):
            _, l, a = self.session.run([self._predictions], {self._X: batch})

    def close(self):
        self.session.close()
