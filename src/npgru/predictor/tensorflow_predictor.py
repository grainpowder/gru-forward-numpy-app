from typing import List, Tuple

import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras as keras

from npgru.predictor.category_predictor import CategoryPredictor
from npgru.preprocessor.model_file import get_model_dir


class TensorflowPredictor(CategoryPredictor):

    def __init__(self):
        model_dir = get_model_dir()
        self._tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir.joinpath("tokenizer.model")))
        self._model = keras.models.load_model(model_dir.joinpath("tensorflow"))

    def predict(self, title: str, num_predictions) -> List[Tuple[int, float]]:
        tokenized_title = self._tokenizer.encode(title) if title else [1]
        probabilities = self._model(tf.constant([tokenized_title]))
        prediction = sorted(enumerate(probabilities.numpy()[0]), key=lambda x: x[1], reverse=True)[:num_predictions]
        return prediction
