from transformers import AutoTokenizer
from datasets import Dataset
import json

from services.tokenisation.abstract_tokeniser import AbstractTokeniser


class HuggingFaceTokeniser(AbstractTokeniser):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize(self, text: str):
        return self.tokenizer(
            text, padding="max_length", truncation=True, max_length=512
        )

    def decode(self, token_ids: list):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize_dataset(self, dataset: Dataset):
        def tokenize_function(examples):
            tokens = self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )
            tokens["labels"] = tokens["input_ids"].copy()  # Use input_ids as labels
            return tokens

        return dataset.map(tokenize_function, batched=True)

    def save_tokenized_data_to_file(self, dataset: Dataset, output_file: str):
        dataset_dict = dataset.to_dict()
        with open(output_file, "w") as f:
            for i in range(len(dataset_dict["input_ids"])):
                tokenized_entry = {
                    "input_ids": dataset_dict["input_ids"][i],
                    "attention_mask": dataset_dict["attention_mask"][i],
                }
                f.write(json.dumps(tokenized_entry) + "\n")

    def read_tokenized_data_from_file(self, input_file: str):
        data = []
        with open(input_file, "r") as f:
            for line in f:
                tokenized_entry = json.loads(line.strip())
                tokenized_entry["labels"] = tokenized_entry["input_ids"].copy()
                data.append(tokenized_entry)
        return Dataset.from_list(data)
