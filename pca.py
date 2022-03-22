import numpy as np
import faiss
from pathlib import Path


class PCA:
    def __init__(self, input_dim, n_components, dataset_name, input_type):
        self.input_dim = input_dim
        self.n_components = n_components
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.filename = f"pca_{self.dataset_name}_{self.input_type}_{self.n_components}.faiss.bin"
        self.faiss_pca = faiss.PCAMatrix(d_in=self.input_dim, d_out=self.n_components)

    def __call__(self, data=None):
        return self.compress(data)

    def train(self, data):
        data = np.ascontiguousarray(data).astype(np.float32)
        self.faiss_pca.train(data)
        return self.faiss_pca

    def compress(self, data):
        if not self.faiss_pca.is_trained:
            self.train(data)
        compressed = self.faiss_pca.apply_py(data)
        return compressed

    def reconstruct(self, data):
        if not self.faiss_pca.is_trained:
            self.train(data)
        data = data.reshape(-1, self.n_components)
        reconstructed = self.faiss_pca.reverse_transform(data)
        return reconstructed

    def save(self, folder_path: Path):
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = folder_path / self.filename
        faiss.write_VectorTransform(self.faiss_pca, str(filename))
        return filename

    @staticmethod
    def load(filepath: Path):
        faiss_pca = faiss.read_VectorTransform(str(filepath))
        return faiss_pca
