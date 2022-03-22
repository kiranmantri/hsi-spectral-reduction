import os
import numpy as np
import logging
import concurrent
import functools
import traceback
import click
from pathlib import Path
from tqdm import tqdm


from compression.hsi_dataset import HSIDataset
from compression.spectral_ae import Autoencoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Init...")


def train(dataset, latent_size):
    name = dataset.name
    logging.info(f"Train AE started: {name}, latent Size:{latent_size}")

    ae = Autoencoder(input_size=dataset.n_bands, latent_size=latent_size, dataset_name=name)
    ae.train(dataset.tf_dataset(batch_size=256), epochs=5)
    logging.info(f"Saving: {name} {latent_size}")
    folder_path = Path(dataset.base_dir)
    filename = ae.save(folder_path=folder_path)
    return filename


@click.command()
@click.option('-b', 'base_dir', required=True, help='Root folder to look for data and also save results')
def main(base_dir):
    datasets = [
        {'file_path': 'data/forest/20170820_Forest_Final_INT.tif', 'name': "forest"},
        {'file_path': 'data/urban1/20170820_Urban_Ref_Reg_Subset.tif', 'name': "urban1"},
        {'file_path': 'data/urban2/20170820_Urban2_INT_Final.tif', 'name': "urban2"},
    ]
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        jobs = []
        for ds_item in datasets:
            name = ds_item['name']
            file_path = Path(base_dir) / ds_item['file_path']
            dataset = HSIDataset(file_path=file_path, name=name)
            _ = dataset.feature_vectors['HSI']
            # for latent_size in [2, 50, 100, 150, 200, 250]:
            for latent_size in np.arange(2, 252):
                f = functools.partial(train, dataset=dataset, latent_size=latent_size)
                job = executor.submit(f)
                jobs.append(job)

        for future in tqdm(concurrent.futures.as_completed(jobs), desc='Training', total=(251 * 3)):
            try:
                data = future.result()
                logging.info(f'File {data} generated')
            except Exception as exc:
                trace = ''.join(traceback.format_tb(exc.__traceback__))
                trace2 = ''.join(traceback.format_tb(future.exception().__traceback__))
                logging.critical(f'Job  generated an exception during copying: {exc}')
                logging.critical(trace)
                logging.critical(trace2)


if __name__ == "__main__":
    main()
