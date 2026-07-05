import tensorflow as tf
import numpy as np


class NN(tf.keras.Model):

    def __init__(self, units=50, eps=1, sigma=1, name="nn", activation=tf.tanh, dtype=tf.float32):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.L2(), dtype=dtype),
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.L2(), dtype=dtype),
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.L2(), dtype=dtype),
                tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(), dtype=dtype),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self.eps = eps
        self.sigma = sigma
        self._name = name
        self._dtype = dtype

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    
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

    def train_batch(self, t, y1, y2, batch_size, nepoch=1000):
        t1, t2 = t.copy(), t.copy()
        N = y1.shape[0]
        train_op = tf.function(self.train_op)
        # loss_op = tf.function(lambda: self.loss_function(y_full, t_full))

        loss = []
        for epoch in range(nepoch):
            idx1 = np.random.choice(N, N, replace=False)
            idx2 = np.random.choice(N, N, replace=False)

            for i in range(N//batch_size):

                batch_idx1 = idx1[batch_size*i: batch_size*(i+1)]
                batch_idx2 = idx2[batch_size*i: batch_size*(i+1)]

                t1_batch = tf.constant(t1[batch_idx1, :], self.dtype)
                t2_batch = tf.constant(t2[batch_idx2, :], self.dtype)
                y1_batch = tf.constant(y1[batch_idx1, :], self.dtype)
                y2_batch = tf.constant(y2[batch_idx2, :], self.dtype)

                t_batch = tf.concat([t1_batch, t2_batch], axis=0)
                y_batch = tf.concat([y1_batch, y2_batch], axis=0)

                loss = train_op(t_batch, y_batch)

            print(epoch, loss.numpy(), flush=True)
            self.save_weights(
                filepath="./checkpoints/"+self.name,
                overwrite=True,
            )
        
            # loss += [loss_op().numpy()]
            # print(epoch, loss[-1])
        return loss

    def restore(self, name=None):
        name = self.name if name is None else name
        self.load_weights("./checkpoints/"+self.name)
