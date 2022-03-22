import numpy as np
from pathlib import Path
import tensorflow as tf
from compression.spectral_ae import Autoencoder


class DAE:
    def __init__(self, input_dim, n_components, dataset_name, input_type, hidden_layers_sizes=(500, 400)):
        self.input_dim = input_dim
        self.n_components = n_components
        self.ae_model = Autoencoder(
            input_size=input_dim,
            latent_size=n_components,
            dataset_name=dataset_name,
            input_type=input_type,
            hidden_layers_sizes=hidden_layers_sizes,
        )
        self.is_trained = False

    def __call__(self, data=None):
        return self.compress(data)

    def train(self, data, batch_size=1024):
        mask = np.abs(np.random.normal(0, 1, size=data.shape)) > 1.9
        noised_data = data.copy()
        noised_data[mask] = 0
        tf_dataset = tf.data.Dataset.from_tensor_slices((noised_data, noised_data))
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.shuffle(batch_size * 100)
        self.ae_model.train(tf_dataset, epochs=30)
        self.is_trained = self.ae_model.hparams["is_trained"]
        return self.ae_model

    def compress(self, data):
        if not self.is_trained:
            self.train(data)
        compressed = self.ae_model.compress(data)
        return compressed

    def reconstruct(self, data):
        if not self.is_trained:
            self.train(data)
        data = data.reshape(-1, self.n_components)
        reconstructed = self.ae_model.reconstruct(data)
        return reconstructed

    def save(self, folder_path: Path):
        folder_path.mkdir(parents=True, exist_ok=True)
        self.ae_model.save(folder_path)

    @classmethod
    def load(cls, filepath):
        ae_model = Autoencoder.load(filepath)
        n_components = ae_model.hparams["latent_size"]
        input_dim = ae_model.hparams["input_size"]
        dataset_name = ae_model.hparams["dataset_name"]
        input_type = ae_model.hparams["input_type"]
        hidden_layers_sizes = ae_model.hparams["hidden_layers_sizes"]
        is_trained = ae_model.hparams["is_trained"]

        self = cls(input_dim, n_components, dataset_name, input_type, hidden_layers_sizes)
        self.ae_model = ae_model
        self.is_trained = is_trained

        return self

