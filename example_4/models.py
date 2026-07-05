import tensorflow as tf
import numpy as np


def jvp(y, x, v):
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class NN(tf.keras.Model):

    def __init__(self, units=200, dim=100, name="nn", activation=tf.tanh, dtype=tf.float32):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation, dtype=dtype),
                tf.keras.layers.Dense(units, activation=activation, dtype=dtype),
                tf.keras.layers.Dense(units, activation=activation, dtype=dtype),
                tf.keras.layers.Dense(dim, dtype=dtype),
            ]
        )
        self.nn.build(input_shape=[None, dim+1])
        self.dim = dim
        self._name = name
        self._dtype = dtype
        self.loss_function = None

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    
    def call(self, t, y):
        s_input = tf.concat([t, y], axis=-1)
        return self.nn(s_input)

    def _loss_function_1(self, t, y):
        # score matching loss
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("score matching loss: sequential training")
        s = self.call(
            tf.reshape(t, [-1, 1]),
            tf.reshape(y, [-1, self.dim]),
        )
        s = tf.reshape(s, y.shape) # shape of [N, M, dim]
        
        s_y = []
        for i in range(self.dim):
            s_y += [tf.gradients(s[..., i:i+1], y)[0][..., i:i+1]]
        s_y = tf.concat(s_y, axis=-1) # shape of [N, M, dim]
        s_y = tf.reduce_sum(s_y, axis=-1)
        loss = tf.reduce_mean(0.5 * tf.reduce_sum(s ** 2, axis=-1) + s_y)
        return tf.reduce_mean(loss)
    
    def _loss_function_2(self, t, y):
        # score matching loss
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("score matching loss: batch jacobian")
        batch_size = y.shape[0]
        num = y.shape[1]
        t = tf.reshape(t, [-1, 1])
        y = tf.reshape(y, [-1, self.dim])

        with tf.GradientTape() as tape:
            tape.watch(y)
            s = self.call(t, y)
        s_y = tape.batch_jacobian(s, y) # shape of [N*M, dim, dim]
        s_y = tf.linalg.trace(s_y) # shape of [N*M]
       
        loss = tf.reduce_mean(0.5 * tf.reduce_sum(s ** 2, axis=-1) + s_y)
        return loss
    
    def _loss_function_3(self, t, y):
        # slice score matching loss when m=1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("slice score matching loss")
        s = self.call(
            tf.reshape(t, [-1, 1]),
            tf.reshape(y, [-1, self.dim]),
        )
        s = tf.reshape(s, y.shape) # shape of [N, M, dim]
        
        
        ## Hutchinson trace estimator
        # Gaussian distribution
        v = tf.random.normal(shape=s.shape) # shape of [N, M, dim]
        # Rademacher distribution
        # v = tf.cast(tf.random.uniform(shape=s.shape) > 0.5, tf.float32) * 2 - 1
        tmp_y = tf.gradients(s, y, grad_ys=v)[0] # shape of [N, M, dim]
        s_y = tf.reduce_sum(tmp_y * v, axis=-1) # shape of [N, M, 1]

        # tmp = tf.reduce_sum(s * v, axis=-1) # shape of [N, M, 1]
        # tmp_y = tf.gradients(tmp, y) # shape of [N, M, dim]
        # s_y = tf.reduce_sum(tmp_y * v, axis=-1) # shape of [N, M]
        
        s_2 = tf.reduce_sum(s ** 2, axis=-1) # shape [N, M]
        loss = tf.reduce_mean(0.5 * s_2 + s_y)
        return loss

    def train_op(self, t, y):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train_batch(self, t, y, batch_size, nepoch=1000):
        # t_full = tf.constant(t, self.dtype)
        # y_full = tf.constant(y, self.dtype)
        
        N = y.shape[0]
        train_op = tf.function(self.train_op)
        # loss_op = tf.function(lambda: self.loss_function(t_full, y_full))

        loss = []
        for epoch in range(nepoch):
            idx = np.random.choice(N, N, replace=False)

            for i in range(N//batch_size):
                batch_idx = idx[batch_size*i: batch_size*(i+1)]
                t_batch = tf.constant(t[batch_idx, :], self.dtype)
                y_batch = tf.constant(y[batch_idx, :], self.dtype)
                current_loss = train_op(t_batch, y_batch)
            self.save_weights(
                filepath="./checkpoints/"+self.name,
                overwrite=True,
            )
        
            # loss += [loss_op().numpy()]
            loss += [current_loss.numpy()]
            print(epoch, loss[-1], flush=True)
        return loss

    def restore(self, name=None):
        name = self.name if name is None else name
        self.load_weights("./checkpoints/"+self.name)
