import sklearn.decomposition
from pathlib import Path
import joblib


class ICA:
    def __init__(self, input_dim, n_components, dataset_name, input_type):
        self.input_dim = input_dim
        self.n_components = n_components
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.fast_ica = sklearn.decomposition.FastICA(n_components=self.n_components)
        self.filename = f"ica_{self.dataset_name}_{self.input_type}_{self.n_components}.pickle.bin"
        self.is_trained = False

    def __call__(self, data=None):
        return self.compress(data)

    def train(self, data):
        self.fast_ica.fit(data)
        self.is_trained = True
        return self.fast_ica

    def compress(self, data):
        if not self.is_trained:
            self.train(data)
        compressed = self.fast_ica.transform(data)
        return compressed

    def reconstruct(self, data):
        if not self.is_trained:
            self.train(data)
        data = data.reshape(-1, self.n_components)
        reconstructed = self.fast_ica.inverse_transform(data)
        return reconstructed

    def save(self, folder_path: Path):
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = folder_path / self.filename
        joblib.dump(self.fast_ica, filename)

    @staticmethod
    def load(filepath: Path):
        fast_ica = joblib.load(filepath)
        return fast_ica
