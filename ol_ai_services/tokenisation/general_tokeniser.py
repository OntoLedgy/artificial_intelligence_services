from tokenisation.abstract_tokeniser import AbstractTokeniser
from datasets import load_dataset


class GeneralTokeniser:
    """General Tokenizer class that accepts different tokenizers."""

    def __init__(
            self,
            tokeniser: AbstractTokeniser):
        self.tokeniser = tokeniser
        self.tokenized_dataset = {}

    def tokenize(self, data_files):
        """
        Tokenize text data from files. The files should be JSONL format with a 'text' field.
        Alternatively, you can modify the dataset format based on how you're loading data.
        """
        # Load the dataset from the file(s) (this assumes JSON format with a 'text' field)
        dataset = load_dataset(
            'json',
            data_files=data_files)

        # Tokenization function
        def tokenize_function(examples):
            text_data = examples['text']

            tokens = self.tokeniser.tokenize(
                text_data
            )
            tokens['labels'] = tokens['input_ids'].copy()  # Use input_ids as labels
            return tokens

        # Apply the tokenizer to the dataset
        self.tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True)

        return self.tokenized_dataset

    def print_tokenized_data(self, split="train", num_samples=5):
        tokenized_data_train = self.tokenized_dataset[split]

        for i in \
                range(
                    min(num_samples,
                        len(tokenized_data_train)
                        )
                ):

            print(f"Sample {i + 1}:")

            token_ids = tokenized_data_train[i]["input_ids"]

            tokens = self.tokeniser.decode(token_ids)

            print(f"Token IDs: {token_ids}")
            print(f"Reconstructed Text: {tokens}")
            print("-" * 50)

    def save_tokenized_data_to_file(
        self, split="train", output_file="tokenized_dataset.json"
    ):
        self.tokeniser.save_tokenized_data_to_file(
            self.tokenized_dataset[split], output_file
        )

    def read_tokenized_data_from_file(self, input_file):
        return self.tokeniser.read_tokenized_data_from_file(input_file)
