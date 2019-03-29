from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
import numpy as np
import tensorflow as tf


def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma


def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))


def neg_log_likelihood(y_true, y_pred, sigma=1.0):
    dist = tf.distributions.Normal(loc=y_true, scale=sigma)
    return K.sum(-dist.log_prob(y_pred))


# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)


class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._trainable_weights.append(prior_params)

        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.output_dim,),
                                       initializer=initializers.normal(stddev=prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return self.kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    import matplotlib.pyplot as plt


    def f(x):
        return x*np.sin(x/10) + 10*np.random.random(len(x)) - 5.0

    xtrain = 100*np.array(sorted(np.random.random(100)))
    ytrain = f(xtrain)
    train_size = len(xtrain)

    batch_size = train_size
    num_batches = train_size / batch_size
    kl_loss_weight = 1.0 / num_batches

    x_in = Input(shape=(1,))
    x = DenseVariational(50, kl_loss_weight=kl_loss_weight, activation='relu')(x_in)
    x = DenseVariational(50, kl_loss_weight=kl_loss_weight, activation='relu')(x)
    x = DenseVariational(1, kl_loss_weight=kl_loss_weight)(x)

    model = Model(x_in, x)

    model.compile(loss=neg_log_likelihood, optimizer='adam', metrics=['mse'])

    for i in range(10):
        model.fit(xtrain, ytrain, batch_size=10, epochs=100, verbose=1)
        print('epoch: {}'.format(i))

    X = np.linspace(0, 100, 1000)
    Y = f(X)

    pred_list=[]
    for j in range(100):
        y_pred = model.predict(X)
        pred_list.append(y_pred)

    ymean = np.mean(pred_list, axis=0)
    std = np.mean(pred_list, axis=0)

    plt.plot(xtrain, ytrain, 'ro')
    plt.plot(X, ymean)
    plt.fill_between(X, ymean.flatten() - std.flatten(), ymean.flatten() + std.flatten(), alpha=.25)
    plt.show()

