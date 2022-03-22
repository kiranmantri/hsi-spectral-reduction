import numpy as np
import sklearn.decomposition
import joblib
from pathlib import Path


class KPCA:
    def __init__(self, input_dim, n_components, dataset_name, input_type):
        self.input_dim = input_dim
        self.n_components = n_components
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.model_kpca = sklearn.decomposition.KernelPCA(
            n_components=n_components, kernel="poly", fit_inverse_transform=True
        )
        self.filename = f"kpca_{self.dataset_name}_{self.input_type}_{self.n_components}.pickle.bin"
        self.is_trained = False

    def __call__(self, data=None):
        return self.compress(data)

    def train(self, data):
        p = 0.05
        mask = np.random.choice([False, True], len(data), p=[1 - p, p])
        data = np.ascontiguousarray(data).astype(np.float32)
        data = data[mask]
        self.model_kpca.fit(data)
        self.is_trained = True
        return self.model_kpca

    def compress(self, data):
        if not self.is_trained:
            self.train(data)
        compressed = self.model_kpca.transform(data)
        return compressed

    def reconstruct(self, data):
        if not self.is_trained:
            self.train(data)
        data = data.reshape(-1, self.n_components)
        reconstructed = self.model_kpca.inverse_transform(data)
        return reconstructed

    def save(self, folder_path: Path):
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = folder_path / self.filename
        joblib.dump(self.model_kpca, filename)

    @staticmethod
    def load(filepath: Path):
        kpca = joblib.load(filepath)
        return kpca
