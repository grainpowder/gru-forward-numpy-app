import gzip
import os
import pathlib
import shutil
import sys
import zipfile

import boto3
from dotenv import load_dotenv

COMPRESSED_FILE_NAMES = ["tokenizer.model.gz", "tensorflow.zip", "weights.zip"]


def get_model_dir() -> pathlib.Path:
    load_dotenv()
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir is None:
        root_dir = [path for path in sys.path if path.endswith("src")]
        assert len(root_dir) > 0, "PYTHONPATH is not defined as source root(src)"
        project_dir = pathlib.Path(root_dir[0]).parent
        model_dir = project_dir.joinpath("model")
    else:
        model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir


class ModelFilePreparer:

    def __init__(self):
        load_dotenv()
        self._model_dir = get_model_dir()
        self._compressed_file_names = COMPRESSED_FILE_NAMES

    def prepare(self):
        """
        Prepare compressed model files in model_dir
        """
        if self.every_compressed_file_exists():
            pass
        elif os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            s3_client = boto3.client("s3")
            s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
            s3_file_prefix = os.environ.get("S3_FILE_PREFIX")
            self._download_compressed_files_from_s3(s3_client, s3_bucket_name, s3_file_prefix)
        else:
            raise FileNotFoundError(f"One of {self._compressed_file_names} does not exist, but cannot access to S3")

    def every_compressed_file_exists(self) -> bool:
        model_dir_file_list = os.listdir(self._model_dir)
        return all([file_name in model_dir_file_list for file_name in self._compressed_file_names])

    def _download_compressed_files_from_s3(self, s3_client, s3_bucket_name, s3_file_prefix) -> None:
        s3_keys = [f"{s3_file_prefix}/{file_name}" for file_name in self._compressed_file_names]
        file_paths = [str(self._model_dir.joinpath(file_name)) for file_name in self._compressed_file_names]
        for s3_key, file_path in zip(s3_keys, file_paths):
            s3_client.download_file(s3_bucket_name, s3_key, file_path)


class ModelFileDecompressor:

    def __init__(self):
        load_dotenv()
        self._model_dir = get_model_dir()
        self._compressed_file_names = COMPRESSED_FILE_NAMES
        self._decompressed_file_names = [name.strip(".gz").strip(".zip") for name in self._compressed_file_names]

    def decompress(self) -> None:
        """
        Decompress every compressed files in model_dir
        """
        if self.every_decompressed_file_exists():
            pass
        else:
            for compressed_name, decompressed_name in zip(self._compressed_file_names, self._decompressed_file_names):
                compressed_file_path = self._model_dir.joinpath(compressed_name)
                decompressed_file_path = self._model_dir.joinpath(decompressed_name)
                if compressed_name.endswith(".gz"):
                    with gzip.open(compressed_file_path, "rb") as source, open(decompressed_file_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                elif compressed_name.endswith(".zip"):
                    zipfile.ZipFile(compressed_file_path).extractall(decompressed_file_path)

    def every_decompressed_file_exists(self) -> bool:
        model_dir_file_list = os.listdir(self._model_dir)
        return all([file_name in model_dir_file_list for file_name in self._decompressed_file_names])
