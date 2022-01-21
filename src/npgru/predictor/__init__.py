from enum import Enum

from npgru.predictor.numpy_predictor import NumpyPredictor
from npgru.predictor.tensorflow_predictor import TensorflowPredictor


class PredictorTypes(str, Enum):
    TENSORFLOW = "tf"
    NUMPY = "np"
