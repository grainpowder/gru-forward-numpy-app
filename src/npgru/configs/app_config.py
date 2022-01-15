import os

import fastapi_plugins
from dotenv import load_dotenv

load_dotenv()


@fastapi_plugins.registered_configuration
class AppSettings(fastapi_plugins.ControlSettings):
    APP_NAME = "npgru-app"
    APP_VERSION = "0.1.1"


@fastapi_plugins.registered_configuration(name="np")
class AppSettingsNumpy(AppSettings):
    S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
    MODEL_URI = f"s3://{S3_BUCKET_NAME}/gru-forward-numpy/weights.zip"
    TOKENIZER_URI = f"s3://{S3_BUCKET_NAME}/gru-forward-numpy/tokenizer.model.gz"


@fastapi_plugins.registered_configuration(name="tf")
class AppSettingsTensorflow(AppSettings):
    S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
    MODEL_URI = f"s3://{S3_BUCKET_NAME}/gru-forward-numpy/tensorflow.zip"
    TOKENIZER_URI = f"s3://{S3_BUCKET_NAME}/gru-forward-numpy/tokenizer.model.gz"


def get_config():
    env_type = os.environ.get("ENV", "tf")
    return fastapi_plugins.get_config(env_type)
