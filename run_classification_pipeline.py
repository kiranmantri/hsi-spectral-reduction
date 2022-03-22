import h5py
import traceback
import numpy as np
import pandas as pd
import json

import sklearn.model_selection
import sklearn.metrics

from pathlib import Path
from tqdm import tqdm

from hsi_deeplearning.classifier import Classifier
from hsi_deeplearning.hsi_dataset import HSIDataset

from compression.pca import PCA
from compression.ica import ICA
from compression.kpca import KPCA
from compression.skl_pca import SKLPCA
from compression.ae import AE
from compression.dae import DAE

from itertools import product

from timeit import default_timer as timer


def evaluate(dataset, input_type, compression_class, repetition, compression_rate):
    r = repetition
    execution_timer = {}
    execution_timer["start"] = timer()
    n_components = int(dataset.n_bands * (1 - compression_rate))

    # print(dataset.name, compression_class.__name__, f"{compression_rate:.2f}")
    if n_components == 0:
        print("Zero components does not make sense...")
        return

    filename = (
        f"prediction_{dataset.name}_{input_type}_{compression_class.__name__}_{r}_{compression_rate:.2f}.h5"
    )
    file_path = Path(filename)
    file_path = Path("/storage/kiran/results/data") / file_path
    if file_path.exists():
        print(f"Results file {file_path} already exists.")
        return

    # Extract data from Dataset raster
    train_categories, train_pixels = dataset.trainingset
    test_categories, test_pixels = dataset.testset

    categories = {**train_categories, **test_categories}
    gt = train_pixels + test_pixels
    y = gt.reshape(dataset.n_pixels)

    execution_timer["load_data"] = timer()

    # Fetch the data based on the input type
    X = getattr(dataset, f"{input_type.lower()}").reshape(dataset.n_pixels, -1)
    X = X.astype("float32")

    print("Split TRAIN/TEST")
    # Split into test and train
    gt_df = pd.DataFrame(y, columns=["category"])
    gt_df = gt_df[gt_df.category > 0]
    train_pixels, test_pixels = sklearn.model_selection.train_test_split(
        gt_df, train_size=0.5, stratify=gt_df.values, random_state=42
    )

    # get the hsi values for the TRAIN pixels
    index_train_pixels = train_pixels[train_pixels.category > 0].index
    index_test_pixels = test_pixels[test_pixels.category > 0].index

    y_train = y[index_train_pixels].ravel()
    y_test = y[index_test_pixels].ravel()

    # IF compression is zero, that means, no compression, i.e. original daa
    # However, DAE will run a model, since the target is also to remove noise
    if (
        (compression_rate == 0)
        and (compression_class.__name__ != "DAE")
        and (compression_class.__name__ != "AE")
    ):
        print(f"No compression...")
        X_train = X[index_train_pixels]
        X_test = reconstructed = X[index_test_pixels]
        n_components = dataset.n_bands
        compression = None
    else:
        print(f"Compressing to {n_components} ({compression_rate*100:.2f}%)")
        compression = compression_class(
            dataset.n_bands, n_components=n_components, dataset_name=dataset.name, input_type=input_type
        )
        print(f"Training...")
        print(f"Original Data: {X.shape}, Masked out bad pixels: {X[~dataset.mask_band.reshape(-1)].shape}")

        data = X[~dataset.mask_band.reshape(-1)]
        compression_model = compression.train(data)
        execution_timer["train_compression"] = timer()

        print(f"Compressing X_train using {compression_model}")
        X_train = compression.compress(X[index_train_pixels])
        X_test = compression.compress(X[index_test_pixels])
        execution_timer["compression"] = timer()

        print(f"Calculating reconstruction using {compression_model}")
        reconstructed = compression.reconstruct(X_test)
        execution_timer["reconstruction"] = timer()

    save_folder_name = Path(f"/storage/kiran/models/{compression_class.__name__}_{r}")
    print(f"Saving model {save_folder_name}")

    if compression is not None:
        compression.save(save_folder_name)

    reconstruction_loss = sklearn.metrics.mean_squared_error(X[index_test_pixels], reconstructed)
    print(
        f"Original:{X[index_test_pixels].shape}, compressed: {X_train.shape}\n"
        f"reconstruction loss: {reconstruction_loss}"
    )

    print("Classification...")
    # Train the model
    classifier = Classifier(dataset.name, dimensions=n_components)
    classifier.train(data=X_train, targets=y_train)
    execution_timer["train_classifier"] = timer()

    print("Evaluation")
    # Predict Evaluate
    predicted, confusion_matrix = classifier.evaluate(X_test, y_test, categories)
    execution_timer["test_classifier"] = timer()

    print("Confusion Matrix:")
    print(confusion_matrix)

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(file_path, "w")
    h5_file.create_dataset("predictions", data=predicted)
    h5_file.create_dataset("targets", data=y_test)
    h5_file.create_dataset("test_samples", data=X[index_test_pixels])
    h5_file.create_dataset("compressed_test_samples", data=X_test)
    h5_file.create_dataset("reconstructed_test_samples", data=reconstructed)

    h5_file.attrs["dataset_name"] = dataset.name
    h5_file.attrs["input_type"] = input_type
    h5_file.attrs["compression_class"] = compression_class.__name__
    h5_file.attrs["n_components"] = n_components
    h5_file.attrs["compression_rate"] = compression_rate
    h5_file.attrs["reconstruction_loss"] = reconstruction_loss
    h5_file.attrs["execution_times"] = json.dumps(execution_timer)
    h5_file.attrs["repetition"] = r

    print(f"Times: {execution_timer}\n\t {np.diff(np.array(list(execution_timer.values())))}")
    print(f"File {file_path} saved")
    h5_file.close()

    return


if __name__ == "__main__":

    tiff_files = {
        "Suburban": "/storage/kiran/data/suburban/20170820_Urban_Ref_Reg_Subset.tif",
        "Urban": "/storage/kiran/data/urban/20170820_Urban2_INT_Final.tif",
        "Forest": "/storage/kiran/data/forest/20170820_Forest_Final_INT.tif",
    }

    datasets = []
    for name, file_path in tiff_files.items():
        dataset = HSIDataset(file_path=file_path, name=name)
        datasets.append(dataset)

    # 3 datasets * 2 input_types * 4 algorithms * 101 compression levels
    # plus
    # 3 datasets * 2 input_types * 2 algorithms * 101 compression levels * 9 repetitions

    # (3*2*4*101)+(3*2*2*101*9) = 13332 jobs
    # (3*2*4*100)+(3*2*2*100*9) = 13200 files

    compression_classes_nn = [AE, DAE]
    compression_classes = [PCA, ICA, KPCA, SKLPCA]

    compression_rates = np.linspace(0, 1, 101)

    input_types = ["HSI", "HSI_SG"]
    repetitions = range(1, 10)

    jobs_nn = list(product(datasets, input_types, compression_classes_nn, repetitions, compression_rates))
    jobs = list(product(datasets, input_types, compression_classes, [1], compression_rates))

    jobs += jobs_nn
    for dataset, input_type, compression_class, repetition, compression_rate in tqdm(jobs):
        print(
            f"{dataset.name} | {input_type} | {compression_class.__name__} | {repetition} | {compression_rate:.2f}"
        )
        try:
            evaluate(dataset, input_type, compression_class, repetition, compression_rate)
        except Exception as e:
            trace = "".join(traceback.format_tb(e.__traceback__))
            print(
                f"Error processing: {dataset.name,  compression_class.__name__, repetition, compression_rate}"
            )
            print(trace)
            print(e)
        print("=" * 80)
