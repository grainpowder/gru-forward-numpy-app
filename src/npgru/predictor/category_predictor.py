from abc import abstractmethod, ABC
from typing import List, Tuple


class CategoryPredictor(ABC):

    @abstractmethod
    def __init__(self):
        """
        Load model files
        """

    @abstractmethod
    def predict(self, title: str, num_predictions: int) -> List[Tuple[int, float]]:
        """
        Define forward process of a model

        :param title: Title of an article to be tokenized
        :param num_predictions: Number of categories to predict on a given title
        :return: List of predicted category and corresponding likelihood score
        """
