from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset

import tiktoken


class AbstractTokeniser(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(
            self,
            text: str):
        pass

    @abstractmethod
    def decode(
            self,
            token_ids: list):
        pass

    @abstractmethod
    def tokenize_dataset(
            self,
            dataset: Dataset):
        pass

    @abstractmethod
    def save_tokenized_data_to_file(
            self,
            dataset: Dataset,
            output_file: str):
        pass

    @abstractmethod
    def read_tokenized_data_from_file(
            self,
            input_file: str):
        pass



