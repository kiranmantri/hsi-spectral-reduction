from collections import OrderedDict
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

# Differences from regular autoencoder:
# - The model estimates parameters of a distribution instead of the signals itself
# - Every input value (in this case bands of a HSI) is described by a PDF
# - The model learn parameters of PDF, e.g., for a prior such as a Gaussian, the model will have to learn mu and sigma


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_size, latent_size, hidden_layers_sizes=[]):
        super(VariationalAutoencoder, self).__init__()

        # Difference between encoder and decoder, is just the input <--> output
        # When encoder, it is input to latent
        # When decoder, it is latent to reconstructed_input
        self.encoder = EncoderDecoder(input_size, latent_size, hidden_layers_sizes)
        self.decoder = EncoderDecoder(latent_size, input_size, hidden_layers_sizes)

    def call(self, input):
        encoded_dist = self.encoder(input)
        decoded_dist = self.decoder(encoded_dist)
        return encoded_dist, decoded_dist

    def train(self, input, **kwargs):
        # TODO: Need define a better loss?
        # Here, we are using a regularizer technique (TF feature) at the end of encoder stage
        # In this case, we are adding a KL divergence from the prior w.r.t encoded q(z) distribution
        # then the final loss just adds the log_prob

        # BUG: Loss is receiving input twice.
        # Need to fix loss here
        # MSE: sqrt(input**2 - input**2)
        def loss(input, output_dist):
            print("Input:", input)
            print("Output Dist shape:", output_dist)
            return -output_dist.log_prob(input)  # + tfp.distributions.kl_divergence()

        # optimizer=tf.optimizers.Adam(learning_rate=1e-3)
        self.compile(optimizer='rmsprop', loss=loss)

        # input and output are the same ...
        # loss will calculate the error between "xhat" (estimated x) and the input
        self.fit(x=input, y=input, **kwargs)


class EncoderDecoder(tfk.layers.Layer):
    def __init__(self, input_size: int, output_size: int, hidden_layers_sizes: list = []):
        super(EncoderDecoder, self).__init__()

        # Notes on TensorFlow probability:
        # If we use Normal(.), tf creates a distribution object with batch_size=len(loc)
        # On the other hannd, if we use Indepedent(Normal(.)), tfp creates a distribution object
        # with event_size=len(loc)
        # We need event_size=len(loc), because we have a HSI with 300 dimensions, independently modeled
        # as gaussians

        # TODO: Try with a Multivariate Normal distribution.
        # The covariance matrix can be computed using
        # cov = tfp.stats.covariance(input, input)

        # self.prior = tfp.distributions.Normal(loc=tf.zeros(output_size), scale=1)
        self.prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(output_size), scale=1), reinterpreted_batch_ndims=1
        )

        # Sort based on size.
        # decoder must be reversed order
        hidden_layers_sizes.sort()
        if input_size < output_size:
            self.layer_type = 'encoder'
            output_activation = None
            reg = tfp.layers.KLDivergenceRegularizer(self.prior, weight=1.0)
        else:
            self.layer_type = 'decoder'
            output_activation = None
            reg = None
            hidden_layers_sizes.reverse()

        # Build Layers ---------------------------------------------------------------------------------------------- #
        layers = OrderedDict()
        layers['cast'] = tfk.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        layers['input'] = tfk.layers.Dense(units=input_size, activation='relu', name='input')

        for size in hidden_layers_sizes:
            name = f'hidden_{self.layer_type}_{size}'
            layers[name] = tfk.layers.Dense(units=size, activation='relu', name=name)

        encoded_shape = tfp.layers.IndependentNormal.params_size(output_size)

        layers['output'] = tfk.layers.Dense(units=encoded_shape, activation=output_activation, name="output")
        layers['output_pdf'] = tfp.layers.IndependentNormal(
            event_shape=output_size, activity_regularizer=reg
        )

        # Build Layers ---------------------------------------------------------------------------------------------- #
        self.layers = layers

    def call(self, input):
        out = input
        for name, layer in self.layers.items():
            # print(self.layer_type, name, out.shape, layer)
            out = layer(out)
        return out
