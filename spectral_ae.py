import h5py
import numpy as np
from pathlib import Path
from collections import OrderedDict
import tensorflow as tf
import tensorflow.keras as tfk


class Autoencoder(tf.keras.Model):
    def __init__(
        self,
        input_size,
        latent_size,
        dataset_name,
        input_type,
        hidden_layers_sizes: tuple = (200, 100, 50, 10),
    ):
        super(Autoencoder, self).__init__()

        self.hparams = {}
        self.hparams["latent_size"] = latent_size
        self.hparams["input_size"] = input_size
        self.hparams["output_size"] = input_size
        self.hparams["hidden_layers_sizes"] = hidden_layers_sizes
        self.hparams["dataset_name"] = dataset_name
        self.hparams["input_type"] = input_type
        self.hparams["description"] = "Autoencoder by Kiran Mantripragada"
        self.hparams["is_trained"] = False

        # Difference between encoder and decoder, is just the input <--> output
        # When encoder, input size goes to latent size
        self.encoder = EncoderDecoder(
            input_size=self.hparams["input_size"],
            output_size=self.hparams["latent_size"],
            hidden_layers_sizes=self.hparams["hidden_layers_sizes"],
        )

        # When decoder, latent size goes to output (which is same as input) size
        self.decoder = EncoderDecoder(
            input_size=self.hparams["latent_size"],
            output_size=self.hparams["input_size"],
            hidden_layers_sizes=self.hparams["hidden_layers_sizes"],
        )

    def get_config(self):
        return {"hparams": self.hparams}

    def from_config(cls, config):
        ae_from_config = cls(
            input_size=config["input_size"],
            latent_size=config["latent_size"],
            dataset_name=config["dataset_name"],
            hidden_layers_sizes=config["hidden_layers_sizes"],
        )
        return ae_from_config

    def call(self, input):
        if isinstance(input, tf.data.Dataset):
            input, output = input
        encoded = self.encoder(input)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def compress(self, data):
        return self.encoder(data).numpy()

    def reconstruct(self, data):
        return self.decoder(data).numpy()

    def __compile__(self, **kwargs):
        # optimizer = 'rmsprop'
        optimizer = tf.optimizers.Adam(learning_rate=1e-2)

        loss = "mse"
        # loss = tfk.losses.MeanSquaredError()

        self.compile(optimizer=optimizer, loss=loss)

    def train(self, tf_dataset, **kwargs):
        self.__compile__()
        callbacks = [tfk.callbacks.EarlyStopping(monitor="loss")]
        if kwargs.get("callbacks", None) is not None:
            callbacks += kwargs.pop("callbacks")

        # input and output are the same ...
        # loss will calculate the error between "xhat" (estimated x) and the input
        fit_history = self.fit(
            tf_dataset,
            #  callbacks=callbacks,
            **kwargs,
        )
        self.hparams["is_trained"] = True
        self.hparams["loss"] = list(fit_history.history["loss"])
        return fit_history

    def save(self, folder_path: Path):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        latent_size = self.hparams["latent_size"]
        dataset_name = self.hparams["dataset_name"]
        input_type = self.hparams["input_type"]
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = folder_path / f"spectral_ae_{dataset_name}_{input_type}_{latent_size}.h5"

        # Save the h5 file !
        print(f"Saving model: {filename}")
        self.save_weights(str(filename))

        # # Now add the hparams into the hdf5 file
        # print(self.hparams)
        model_h5_file = h5py.File(filename, "r+")
        model_h5_file.create_group("hparams")
        for key, value in self.hparams.items():
            model_h5_file["hparams"][key] = value
        model_h5_file.close()
        return filename

    @classmethod
    def load(cls, filepath: Path):
        # Load hparams first, before constructing a new Model
        model_h5_file = h5py.File(filepath, "r")
        input_size = model_h5_file["hparams/input_size"][()]
        latent_size = model_h5_file["hparams/latent_size"][()]
        hidden_layers_sizes = model_h5_file["hparams/hidden_layers_sizes"][()]
        dataset_name = model_h5_file["hparams/dataset_name"][()]
        input_type = model_h5_file["hparams/input_type"][()]

        self = cls(
            input_size=input_size,
            latent_size=latent_size,
            dataset_name=dataset_name,
            hidden_layers_sizes=hidden_layers_sizes,
            input_type=input_type,
        )

        def set_hparams(key, value):
            self.hparams[key] = value

        model_h5_file["hparams"].visititems(lambda k, v: set_hparams(k, v[()]))
        model_h5_file.close()

        rand_input = np.random.rand(input_size).reshape(-1, input_size).astype(np.float32)
        self.__compile__()
        _ = self.evaluate(rand_input, rand_input, verbose=0)
        self.load_weights(str(filepath))
        return self


class EncoderDecoder(tfk.layers.Layer):
    def __init__(self, input_size: int, output_size: int, hidden_layers_sizes: tuple = ()):
        super(EncoderDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_sizes = list(hidden_layers_sizes)

        # Sort based on size.
        # decoder must be reversed order

        self.hidden_layers_sizes.reverse()
        if self.input_size > self.output_size:
            self.layer_type = "encoder"
            self.output_activation = "sigmoid"
            name_first_layer = f"encoder_input"
            self.hidden_layers_sizes.reverse()
        else:
            self.layer_type = "decoder"
            self.output_activation = "sigmoid"
            name_first_layer = f"decoder_input"

        # print(self.input_size, self.output_size)
        # print(self.layer_type, self.hidden_layers_sizes)
        # print('----')

        # Build Layers ---------------------------------------------------------------------------------------------- #
        layers = OrderedDict()
        layers["cast"] = tfk.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        layers[name_first_layer] = tfk.layers.Dense(
            name=name_first_layer, units=self.input_size, activation="relu"
        )

        for size in self.hidden_layers_sizes:
            name = f"hidden_{self.layer_type}_{size}"
            layers[name] = tfk.layers.Dense(name=name, units=size, activation="relu")
            # layers[f"{name}_dropout"] = tfk.layers.Dropout(0.2)

        layers[f"output_{self.layer_type}"] = tfk.layers.Dense(
            name=f"output_{self.layer_type}",
            units=self.output_size,
            activation=self.output_activation,
        )
        self.layer_names = list(layers.keys())
        # Build Layers ---------------------------------------------------------------------------------------------- #
        self.layers = layers

    def call(self, input):
        out = input
        for name, layer in self.layers.items():
            # units = layer.units if "units" in layer.__dict__ else "N/A"
            # print(name, layer, units, out.shape)
            # print('-'*70)
            out = layer(out)
        return out

    def get_config(self):
        return dict(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_sizes=self.hidden_layers_sizes,
            activation=self.output_activation,
            layer_names=self.layer_names,
        )
