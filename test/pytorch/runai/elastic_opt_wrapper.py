import sys

import keras.optimizers
import keras.backend as K

import runai.utils

from . import hooks

class Optimizer(keras.optimizers.Optimizer):
    def __init__(self, optimizer, steps):
        super(Optimizer, self).__init__()
        self.optimizer = optimizer
        self.steps = steps

        runai.utils.log.debug('Wrapping \'%s\' Keras optimizer with GA of %d steps', optimizer.__class__.__name__, steps)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        with K.name_scope("runai") as name_scope:
            iterations = K.variable(0, dtype='int64', name='iterations')
            first      = K.equal(iterations % self.steps, 0)
            last       = K.equal(iterations % self.steps, self.steps - 1)

            # variables to hold the accumulated gradients between steps
            vagrads = [K.zeros(K.int_shape(param), dtype=K.dtype(param)) for param in params]

            # reset the accumulated gradient every first iteration
            agrads = [K.switch(first, grad, grad + vagrad) for grad, vagrad in zip(grads, vagrads)]

        with hooks.get_gradients(self.optimizer, [agrad / self.steps for agrad in agrads]), \
            hooks.update    (condition=last, name_scope=name_scope),    \
            hooks.update_add(condition=last, name_scope=name_scope),    \
            hooks.update_sub(condition=last, name_scope=name_scope):
            for update in self.optimizer.get_updates(loss, params):
                # get_updates() may return assignment ops or tuples (variable, gradient) representing the desired assignments
                # in the latter case we want to build the assignment op ourselves under the same hooks
                #
                # reference: https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/backend.py#L3145
                if isinstance(update, tuple):
                    with K.name_scope(name_scope):
                        update = K.update(update[0], update[1])

                self.updates.append(update)

        assert K.backend() == 'tensorflow', "Unsupported backend (" + K.backend() + ")"

        with K.name_scope(name_scope):
            self.updates.extend([K.update(vagrad, agrad) for vagrad, agrad in zip(vagrads, agrads)])

            with K.get_session().graph.control_dependencies(self.updates):
                self.updates.append(K.update_add(iterations, 1))

        return self.updates

    def get_gradients(self, loss, params):
        return self.optimizer.get_gradients(loss, params)

    def set_weights(self, weights):
        self.optimizer.set_weights(weights)

    def get_weights(self):
        return self.optimizer.get_weights()

    def get_config(self):
        # we have to support creating our optimizers from configurations in order to support being run with Horovod
        # Horovod dynamically creates a class that inherits the optimizer class it's wrapping (our optimizers), and
        # passes the dictionary returned from this very method as the kwargs for the initialization in __init__()
        #
        # our optimizers inherit from this very class, receive 'steps' as an argument, and do not receive 'optimizer'
        # as they create the one they mimic
        #
        # therefore, we do not save self.optimizer in the returned dictionary

        config = self.optimizer.get_config()
        config['steps'] = self.steps
        return config

    def __getattribute__(self, name):
        # users can query the optimizer to retrieve its attributes, such as 'lr'.
        # we rely on the fact that there are no mutual attribute names between our
        # implementation and the original optimizer implementation, and we get the
        # original optimizer's attribute in case our object does not have one.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return self.optimizer.__getattribute__(name)

def _optimizer(optimizer):
    setattr(
        sys.modules[__name__],
        optimizer,
        type(
            optimizer, (Optimizer,),
            { '__init__': lambda self, steps, **kwargs: Optimizer.__init__(self, optimizer=getattr(keras.optimizers, optimizer)(**kwargs), steps=steps) }
        )
    )

[_optimizer(optimizer) for optimizer in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']]