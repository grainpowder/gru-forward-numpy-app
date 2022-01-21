from typing import List, Tuple

import numpy as np
import pandas as pd
import sentencepiece as spm
from scipy.special import softmax

from npgru.predictor.category_predictor import CategoryPredictor
from npgru.preprocessor.model_file import get_model_dir


class NumpyPredictor(CategoryPredictor):

    def __init__(self):
        model_dir = get_model_dir()
        weight_dir = model_dir.joinpath("weights")
        self._tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir.joinpath("tokenizer.model")))
        self._embedding_affine = pd.read_csv(weight_dir.joinpath("embedding_affine.csv"), header=None).values
        self._hidden_bias = pd.read_csv(weight_dir.joinpath("hidden_bias.csv"), header=None).values.squeeze()
        self._hidden_kernel = pd.read_csv(weight_dir.joinpath("hidden_kernel.csv"), header=None).values
        self._hidden_dim = self._hidden_kernel.shape[0]
        self._dense_bias = pd.read_csv(weight_dir.joinpath("dense_bias.csv"), header=None).values.squeeze()
        self._dense_kernel = pd.read_csv(weight_dir.joinpath("dense_kernel.csv"), header=None).values

    def predict(self, title: str, num_predictions: int) -> List[Tuple[int, float]]:
        tokenized_title = self._tokenizer.encode(title) if title else [1]
        hidden = np.zeros(self._hidden_dim, dtype=float)
        for token in tokenized_title:
            hidden = self._calculate_next_hidden(token, hidden)
        logits = (hidden * (hidden > 0)) @ self._dense_kernel + self._dense_bias
        probabilities = softmax(logits)
        prediction = [(int(index), probabilities[index])
                      for index
                      in np.argsort(-logits)[:num_predictions]]
        return prediction

    def _calculate_next_hidden(self, current_token: int, previous_hidden: np.array) -> np.array:
        hidden_dim = self._hidden_dim
        transformed_embedding = self._embedding_affine[current_token, :]
        transformed_hidden = previous_hidden @ self._hidden_kernel + self._hidden_bias
        update_operand = transformed_embedding[:hidden_dim] + transformed_hidden[:hidden_dim]
        reset_operand = transformed_embedding[hidden_dim:(2 * hidden_dim)] + \
                        transformed_hidden[hidden_dim:(2 * hidden_dim)]

        update_gate = 1 / (1 + np.exp(-update_operand))  # apply sigmoid
        reset_gate = 1 / (1 + np.exp(-reset_operand))
        candidate_operand = transformed_embedding[(2 * hidden_dim):] + \
                            reset_gate * transformed_hidden[(2 * hidden_dim):]
        candidate_hidden = np.tanh(candidate_operand)

        return (1 - update_gate) * candidate_hidden + update_gate * previous_hidden
