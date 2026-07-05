import tensorflow as tf
import numpy as np


class NN(tf.keras.Model):

    def __init__(self, units=50, name="nn", activation=tf.tanh, dtype=tf.float32):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(1),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self._name = name
        self._dtype = dtype

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def call(self, t, y):
        s_input = tf.concat([t, y], axis=-1)
        return self.nn(s_input) 

    def loss_function(self, t, y):
        # slice score matching loss
        s = self.call(
            tf.reshape(t, [-1, 1]),
            tf.reshape(y, [-1, 1]),
        )
        s = tf.reshape(s, y.shape)
        s_y = tf.gradients(s, y)[0]
        loss = tf.reduce_mean(0.5 * s ** 2 + s_y)
        return tf.reduce_mean(loss)

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
                loss_value = train_op(t_batch, y_batch)
            self.save_weights(
                filepath="./checkpoints/"+self.name,
                overwrite=True,
            )
        
            # loss += [loss_op().numpy()]
            loss += [loss_value.numpy()]
            print(epoch, loss[-1])
        return loss

    def restore(self, name=None):
        name = self.name if name is None else name
        self.load_weights("./checkpoints/"+self.name)
