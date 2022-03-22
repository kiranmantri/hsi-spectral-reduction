import h5py
import traceback
import numpy as np
import pandas as pd
import json

import sklearn.model_selection
import sklearn.metrics

from pathlib import Path
from tqdm.auto import tqdm

from compression.classifier import Classifier
from compression.hsi_dataset import HSIDataset
from itertools import product
from timeit import default_timer as timer


def classification_rgb(
    dataset, input_type,
):
    execution_timer = {}
    execution_timer["start"] = timer()
    print(dataset.name, input_type)

    filename = f"classification_baseline_{dataset.name}_{input_type}.h5"
    file_path = Path(filename)
    file_path = Path("/storage/kiran/results/data") / file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        print(f"Results file {file_path} already exists.")
        return

    # Extract data from Dataset raster
    train_categories, train_pixels = dataset.trainingset
    test_categories, test_pixels = dataset.testset

    categories = {**train_categories, **test_categories}
    gt = train_pixels + test_pixels
    y = gt.reshape(dataset.n_pixels)

    # Fetch the data based on the input type
    #  X = dataset.rgb.reshape(dataset.n_pixels, -1)
    X = getattr(dataset, f"{input_type.lower()}").reshape(dataset.n_pixels, -1)
    n_components = X.shape[1]
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

    X_train = X[index_train_pixels]
    X_test = X[index_test_pixels]

    print("Classification...")
    # Train the model
    classifier = Classifier(dataset.name, dimensions=3)
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
    h5_file.create_dataset("compressed_test_samples", data=np.array([]))
    h5_file.create_dataset("reconstructed_test_samples", data=np.array([]))

    h5_file.attrs["dataset_name"] = dataset.name
    h5_file.attrs["input_type"] = input_type
    h5_file.attrs["compression_class"] = "NA"
    h5_file.attrs["n_components"] = n_components
    h5_file.attrs["compression_rate"] = 0
    h5_file.attrs["reconstruction_loss"] = 0
    h5_file.attrs["execution_times"] = json.dumps(execution_timer)
    h5_file.attrs["repetition"] = 1

    print("Times:", execution_timer)

    print(f"File {file_path} saved")
    h5_file.close()

    return predicted, confusion_matrix


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

    input_types = ["HSI", "HSI_SG", "RGB"]

    jobs = product(datasets, input_types)
    for dataset, input_type in tqdm(jobs):
        try:
            print(dataset.name, input_type)
            classification_rgb(dataset, input_type)
        except Exception as e:
            trace = "".join(traceback.format_tb(e.__traceback__))
            print(f"Error processing: {dataset.name}")
            print(trace)
            print(e)
        print("=" * 80)
