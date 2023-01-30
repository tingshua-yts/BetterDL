def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params) # 计算梯度
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
        lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                    K.dtype(self.decay))))
    # momentum
    shapes = [K.int_shape(p) for p in params]
    moments = [K.zeros(shape) for shape in shapes]
    self.weights = [self.iterations] + moments
    for p, g, m in zip(params, grads, moments):
        v = self.momentum * m - lr * g  # velocity
        self.updates.append(K.update(m, v))

        if self.nesterov:
            new_p = p + self.momentum * v - lr * g
        else:
            new_p = p + v

        # Apply constraints.
        if getattr(p, 'constraint', None) is not None:
            new_p = p.constraint(new_p)

        self.updates.append(K.update(p, new_p))
    return self.updates
