import tiktoken
import json

from services.tokenisation.abstract_tokeniser import AbstractTokeniser
from datasets import Dataset, load_dataset


class OpenAiTokeniser(AbstractTokeniser):
    def __init__(self, model_name: str):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str):
        return {"input_ids": self.encoding.encode(text)}

    def decode(self, token_ids: list):
        return self.encoding.decode(token_ids)

    def tokenize_dataset(self, dataset: Dataset):
        def tokenize_function(examples):
            tokens = self.tokenize(examples["text"])
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        return dataset.map(tokenize_function, batched=True)

    def save_tokenized_data_to_file(self, dataset: Dataset, output_file: str):
        dataset_dict = dataset.to_dict()
        with open(output_file, "w") as f:
            for i in range(len(dataset_dict["input_ids"])):
                tokenized_entry = {"input_ids": dataset_dict["input_ids"][i]}
                f.write(json.dumps(tokenized_entry) + "\n")

    def read_tokenized_data_from_file(self, input_file: str):
        data = []
        with open(input_file, "r") as f:
            for line in f:
                tokenized_entry = json.loads(line.strip())
                tokenized_entry["labels"] = tokenized_entry["input_ids"].copy()
                data.append(tokenized_entry)
        return Dataset.from_list(data)
