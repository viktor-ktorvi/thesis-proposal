import gdown
import os
import tarfile

from pathlib import Path


def download(path: str, google_drive_url: str, quiet: bool = False):
    """
    Downloads a dataset from the Google Drive URL into the given directory if the given directory doesn't already exist and is not empty.

    :param path: Directory where the data will be stored.
    :param google_drive_url: Google Drive download link.
    :param quiet: Whether to display progress while downloading. True will supress the output. False by default.
    :return:
    """
    compressed_filepath = path + "tar.gz"

    if not os.path.isdir(path) or not os.listdir(path):  # if dir doesn't exist or is empty
        Path(path).mkdir(parents=True, exist_ok=True)
        gdown.download(google_drive_url, compressed_filepath, quiet=quiet, fuzzy=True)

        # extract
        with tarfile.open(compressed_filepath, "r:gz") as f:
            f.extractall(path=path)

        os.remove(compressed_filepath)
